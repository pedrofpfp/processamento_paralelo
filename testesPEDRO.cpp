// =============================================================================
// ANALISADOR CSV PARALELO — OpenMP + mmap + Reservoir Sampling
// Compilação: g++ -O3 -fopenmp -o analisador final.cpp
// Uso:        ./analisador dados.csv [delimitador]
//
// Estratégia de memória:
//   - mmap: SO gerencia paginação, nunca carrega o arquivo inteiro na RAM
//   - Chunks: processa N linhas por vez, descarta após merge
//   - Reservoir sampling: amostra aleatória de tamanho fixo para percentis
//     (sem guardar todos os valores — funciona com qualquer tamanho de arquivo)
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// =============================================================================
// CONFIGURAÇÕES — ajuste conforme RAM disponível
// =============================================================================
static const size_t CHUNK_LINES = 1000'000; // linhas por chunk (RAM controlada)
static const size_t RES_LOCAL   =  5'000; // reservoir por thread por coluna
static const size_t RES_GLOBAL  = 50'000; // reservoir global por coluna (~2.4 MB/col)

// =============================================================================
// UTILITÁRIOS (sem <algorithm>)
// =============================================================================

static void trim(const char*& s, size_t& len) {
    while (len && (s[0]==' '||s[0]=='\t'||s[0]=='\r')) { s++; len--; }
    while (len && (s[len-1]==' '||s[len-1]=='\t'||s[len-1]=='\r')) len--;
}

static bool to_double(const char* s, size_t len, double& out) {
    if (!len) return false;
    char* e; out = strtod(s, &e);
    return e == s + len;
}

static bool is_null(const char* s, size_t len) {
    return !len
        || (len==2 && s[0]=='N' && s[1]=='A')
        || (len==4 && !strncmp(s,"null",4))
        || (len==3 && !strncmp(s,"N/A",3))
        || (len==3 && !strncmp(s,"nan",3));
}

// Gerador pseudo-aleatório por thread (sem rand() global — thread-safe)
static uint64_t lcg(uint64_t& st) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
}

// Quicksort para ordenar o reservoir e calcular percentis
static void qsort_d(std::vector<double>& v, int lo, int hi) {
    if (hi - lo < 2) return;
    if (hi - lo <= 32) { // insertion sort para arrays pequenos
        for (int i = lo+1; i < hi; i++) {
            double x = v[i]; int j = i-1;
            while (j >= lo && v[j] > x) { v[j+1]=v[j]; j--; }
            v[j+1] = x;
        }
        return;
    }
    double p = v[lo + (hi-lo)/2]; int i=lo, j=hi-1;
    while (i<=j) {
        while (v[i]<p) i++; while (v[j]>p) j--;
        if (i<=j) { double t=v[i]; v[i]=v[j]; v[j]=t; i++; j--; }
    }
    #pragma omp task shared(v) firstprivate(lo,j) if(hi-lo > 50000)
    qsort_d(v, lo, j+1);
    #pragma omp task shared(v) firstprivate(i,hi) if(hi-lo > 50000)
    qsort_d(v, i, hi);
    #pragma omp taskwait
}

// =============================================================================
// ESTRUTURAS DE DADOS
// =============================================================================

// Dicionário string→int para colunas categóricas
struct Dict {
    std::unordered_map<std::string,int> s2i;
    std::vector<std::string>            i2s;
    std::vector<size_t>                 cnt;

    void add(const char* s, size_t len, size_t n = 1) {
        std::string k(s, len);
        auto it = s2i.find(k);
        if (it == s2i.end()) {
            s2i[k] = (int)i2s.size();
            i2s.push_back(k); cnt.push_back(n);
        } else { cnt[it->second] += n; }
    }
};

// ---------------------------------------------------------------------------
// LCol — estado LOCAL de uma coluna dentro de UMA thread em UM chunk
//
// Por que local? Cada thread escreve em seu próprio LCol → zero locks.
// O merge (agregação) é feito depois, por coluna, em paralelo.
// ---------------------------------------------------------------------------
struct LCol {
    bool   num = true, first = true;
    size_t nulos = 0, count = 0, seen = 0;
    double mean = 0, M2 = 0, vmin = 0, vmax = 0;
    std::vector<double> res;   // reservoir amostral (tamanho fixo)
    uint64_t rng = 0;
    Dict dict;

    void init(int tid, int cid) {
        res.reserve(RES_LOCAL);
        rng = (uint64_t)tid * 2654435761ULL ^ (uint64_t)cid * 40503ULL;
    }

    // Reset para próximo chunk, herda tipo global já determinado
    void reset(bool tipo_global) {
        num=tipo_global; first=true; nulos=0; count=0; seen=0;
        mean=0; M2=0; vmin=0; vmax=0;
        res.clear();
        dict.s2i.clear(); dict.i2s.clear(); dict.cnt.clear();
    }

    // Adiciona valor numérico: Welford + reservoir sampling
    void add_num(double x) {
        count++; seen++;
        if (first||x<vmin) vmin=x;
        if (first||x>vmax) vmax=x;
        first = false;
        // Welford: média e variância em uma passagem, numericamente estável
        double d = x - mean; mean += d/count; M2 += d*(x-mean);
        // Reservoir (Algorithm R): mantém amostra uniforme sem guardar tudo
        if (res.size() < RES_LOCAL) res.push_back(x);
        else { size_t j = lcg(rng)%seen; if (j < RES_LOCAL) res[j] = x; }
    }

    // Coluna descoberta como categórica: descarta estado numérico, troca modo
    void vira_cat(const char* s, size_t len) {
        num=false; count=0; mean=0; M2=0; vmin=0; vmax=0; res.clear();
        dict.add(s, len);
    }
};

// ---------------------------------------------------------------------------
// GCol — estado GLOBAL acumulado de uma coluna (todos os chunks, todas as threads)
// ---------------------------------------------------------------------------
struct GCol {
    std::string nome;
    bool   num = true, first = true;
    size_t nulos = 0, count = 0, seen_res = 0;
    double mean = 0, M2 = 0, vmin = 0, vmax = 0;
    std::vector<double> res; // reservoir global (tamanho fixo = RES_GLOBAL)
    uint64_t rng = 0;
    Dict dict;
    std::ostringstream out;

    void init(const std::string& n, int idx) {
        nome = n; res.reserve(RES_GLOBAL);
        rng = (uint64_t)idx * 1234567891ULL ^ 0xCAFEBABEULL;
    }

    // Combina estado Welford de uma thread (fórmula de Chan et al. — exata)
    void merge_num(const LCol& l) {
        if (!l.count) return;
        if (first || l.vmin < vmin) vmin = l.vmin;
        if (first || l.vmax > vmax) vmax = l.vmax;
        first = false;
        size_t n = count + l.count;
        double delta = l.mean - mean;
        mean = (mean*count + l.mean*l.count) / (double)n;
        M2  += l.M2 + delta*delta*(double)count*(double)l.count/(double)n;
        count = n;
        // Insere amostras do reservoir local no global (mantém limite RES_GLOBAL)
        for (double x : l.res) {
            seen_res++;
            if (res.size() < RES_GLOBAL) res.push_back(x);
            else { size_t j = lcg(rng)%seen_res; if (j < RES_GLOBAL) res[j] = x; }
        }
    }

    void merge_cat(const LCol& l) {
        for (size_t i = 0; i < l.dict.i2s.size(); i++)
            dict.add(l.dict.i2s[i].c_str(), l.dict.i2s[i].size(), l.dict.cnt[i]);
    }
};

struct Ctx {
    std::vector<GCol> cols;
    size_t total = 0;
    double t_io = 0, t_proc = 0, t_stats = 0;
    std::chrono::high_resolution_clock::time_point t0;
};

// =============================================================================
// ETAPA 1 — mmap: mapeia arquivo na memória sem cópia
// O SO lê do disco sob demanda e libera páginas antigas conforme avançamos.
// MADV_SEQUENTIAL: pré-busca agressiva (ideal para leitura sequencial).
// =============================================================================
struct Mapa { const char* buf=nullptr; size_t tam=0; int fd=-1; };

static bool open_mmap(const std::string& f, Mapa& m) {
    m.fd = open(f.c_str(), O_RDONLY);
    if (m.fd < 0) { std::cerr << "ERRO: " << f << "\n"; return false; }
    struct stat st; fstat(m.fd, &st); m.tam = st.st_size;
    m.buf = (const char*)mmap(nullptr, m.tam, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.buf == MAP_FAILED) { std::cerr << "ERRO mmap\n"; close(m.fd); return false; }
    madvise((void*)m.buf, m.tam, MADV_SEQUENTIAL);
    return true;
}
static void close_mmap(Mapa& m) { munmap((void*)m.buf, m.tam); close(m.fd); }

// =============================================================================
// ETAPA 2 — Lê cabeçalho e inicializa colunas
// =============================================================================
static size_t init_cols(const Mapa& m, char delim, Ctx& ctx) {
    size_t p = 0;
    while (p < m.tam && m.buf[p] != '\n') p++;
    size_t hend = p; if (p < m.tam) p++;

    int idx = 0; size_t s = 0;
    for (size_t i = 0; i <= hend; i++) {
        if (i == hend || m.buf[i] == delim) {
            const char* ns = m.buf+s; size_t nl = i-s;
            trim(ns, nl);
            GCol c; c.init(std::string(ns,nl), idx++);
            ctx.cols.push_back(std::move(c));
            s = i+1;
        }
    }
    return p;
}

// =============================================================================
// ETAPA 3 — Processar em chunks (memória controlada)
//
// Loop externo: sequencial — coleta CHUNK_LINES offsets de linha de uma vez.
// Loop interno: paralelo por linha — cada thread tem seus LCols locais.
// Merge por coluna: paralelo, sem locks.
//
// Consumo de RAM por chunk (aprox.):
//   offsets:    CHUNK_LINES × 8 bytes   = ~2.4 MB
//   LCols:      threads × cols × (reservoir local) = controlado
// =============================================================================
static void processar(const Mapa& m, size_t pos0, char delim, Ctx& ctx) {
    int nt = omp_get_max_threads();
    int NC = (int)ctx.cols.size();
    size_t pos = pos0;

    // Aloca LCols uma vez, reutiliza entre chunks (evita malloc por chunk)
    std::vector<std::vector<LCol>> local(nt, std::vector<LCol>(NC));
    for (int t = 0; t < nt; t++)
        for (int c = 0; c < NC; c++)
            local[t][c].init(t, c);

    while (pos < m.tam) {
        // --- Coleta offsets do chunk (sequencial, muito rápido) ---
        std::vector<size_t> offs;
        offs.reserve(CHUNK_LINES);
        size_t p = pos;
        while (p < m.tam && offs.size() < CHUNK_LINES) {
            offs.push_back(p);
            while (p < m.tam && m.buf[p] != '\n') p++;
            if (p < m.tam) p++;
        }
        pos = p;
        size_t NL = offs.size();
        ctx.total += NL;

        // Propaga tipo global para os LCols (coluna já definida como categórica
        // não volta a ser numérica em chunks futuros)
        for (int t = 0; t < nt; t++)
            for (int c = 0; c < NC; c++)
                local[t][c].reset(ctx.cols[c].num);

        // --- Processa linhas do chunk em paralelo (por linha, lock-free) ---
        #pragma omp parallel for schedule(dynamic, 2000) num_threads(nt)
        for (size_t li = 0; li < NL; li++) {
            int    tid = omp_get_thread_num();
            size_t p0  = offs[li];
            size_t p1  = (li+1 < NL) ? offs[li+1] : (pos < m.tam ? pos : m.tam);
            int    col = 0; size_t f = p0;

            while (f < p1 && col < NC) {
                size_t g = f;
                while (g < p1 && m.buf[g] != delim && m.buf[g] != '\n' && m.buf[g] != '\r') g++;
                const char* sv = m.buf+f; size_t sl = g-f;
                trim(sv, sl);

                LCol& lc = local[tid][col];
                if (is_null(sv, sl)) {
                    lc.nulos++;
                } else if (lc.num) {
                    double d;
                    if (to_double(sv, sl, d)) lc.add_num(d);
                    else                      lc.vira_cat(sv, sl);
                } else {
                    lc.dict.add(sv, sl);
                }
                f = g+1; col++;
            }
        }

        // --- Merge local → global (paralelo por coluna, sem locks) ---
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (int col = 0; col < NC; col++) {
            GCol& gc = ctx.cols[col];
            // Determina tipo deste chunk: se qualquer thread viu não-numérico → categórica
            for (int t = 0; t < nt; t++)
                if (!local[t][col].num) { gc.num = false; break; }

            gc.nulos += [&]{ size_t s=0; for (int t=0;t<nt;t++) s+=local[t][col].nulos; return s; }();

            if (gc.num) {
                for (int t = 0; t < nt; t++) gc.merge_num(local[t][col]);
            } else {
                for (int t = 0; t < nt; t++) gc.merge_cat(local[t][col]);
            }
        }

        std::cout << "> " << ctx.total << " linhas...\r" << std::flush;
    }
    std::cout << "\n";
}

// =============================================================================
// ETAPA 4 — Calcula e formata estatísticas (uma task por coluna)
// =============================================================================

static void stats_num(GCol& col) {
    auto& o = col.out;
    size_t total = col.count + col.nulos;
    o << "\n[NUM] " << col.nome << "\n"
      << "  Nulos   : " << col.nulos << " ("
      << std::fixed << std::setprecision(2)
      << (total ? col.nulos*100.0/total : 0) << "%)\n";
    if (!col.count) { o << "  (sem dados)\n"; return; }

    double desvio = (col.count > 1 && col.M2 > 0) ? std::sqrt(col.M2/col.count) : 0;

    // Ordena reservoir com tasks OpenMP para percentis aproximados
    #pragma omp taskgroup
    { qsort_d(col.res, 0, (int)col.res.size()); }

    size_t nr = col.res.size();
    double p25 = nr ? col.res[nr/4]   : 0;
    double med = nr ? (nr%2==0 ? (col.res[nr/2-1]+col.res[nr/2])*0.5 : col.res[nr/2]) : 0;
    double p75 = nr ? col.res[3*nr/4] : 0;

    o << "  Registros: " << col.count << "\n"
      << "  Min      : " << col.vmin  << "\n"
      << "  Max      : " << col.vmax  << "\n"
      << "  Media    : " << col.mean  << "\n"
      << "  Mediana* : " << med << "  (*amostral, " << nr << " pontos)\n"
      << "  P25/P75* : " << p25 << " / " << p75 << "\n"
      << "  Desvio   : " << desvio    << "\n";
}

static void stats_cat(GCol& col) {
    auto& o = col.out;
    size_t total_v = 0;
    for (size_t c : col.dict.cnt) total_v += c;
    size_t total = total_v + col.nulos;

    o << "\n[CAT] " << col.nome << "\n"
      << "  Nulos   : " << col.nulos << " ("
      << std::fixed << std::setprecision(2)
      << (total ? col.nulos*100.0/total : 0) << "%)\n"
      << "  Unicos  : " << col.dict.i2s.size() << "\n";

    if (!total_v) return;
    const int TOP = 5;
    std::vector<bool> used(col.dict.i2s.size(), false);
    o << "  Top " << TOP << ":\n";
    for (int k = 0; k < TOP && k < (int)col.dict.i2s.size(); k++) {
        size_t best = col.dict.i2s.size();
        for (size_t i = 0; i < col.dict.i2s.size(); i++)
            if (!used[i] && (best==col.dict.i2s.size() || col.dict.cnt[i]>col.dict.cnt[best]))
                best = i;
        if (best == col.dict.i2s.size()) break;
        double f = col.dict.cnt[best]*100.0/total_v;
        o << "    \"" << col.dict.i2s[best] << "\" — "
          << col.dict.cnt[best] << " (" << f << "%)\n";
        used[best] = true;
    }
}

static void calc_stats(Ctx& ctx) {
    int N = (int)ctx.cols.size();
    #pragma omp parallel
    #pragma omp single nowait
    for (int i = 0; i < N; i++) {
        #pragma omp task firstprivate(i) shared(ctx)
        { if (ctx.cols[i].num) stats_num(ctx.cols[i]); else stats_cat(ctx.cols[i]); }
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    const std::string ARQ   = (argc > 1) ? argv[1] : "dados.csv";
    const char        DELIM = (argc > 2) ? argv[2][0] : ',';

    auto now  = []{ return std::chrono::high_resolution_clock::now(); };
    auto secs = [](auto d){ return std::chrono::duration<double>(d).count(); };

    std::cout << "Threads : " << omp_get_max_threads() << "\n"
              << "Arquivo : " << ARQ << "\n\n";

    Ctx ctx; ctx.t0 = now();

    // Etapa 1: mmap
    Mapa m;
    if (!open_mmap(ARQ, m)) return 1;
    std::cout << "Tamanho : " << m.tam/(1024*1024) << " MB\n";

    // Etapa 2: cabeçalho
    size_t pos_dados = init_cols(m, DELIM, ctx);
    std::cout << "Colunas : " << ctx.cols.size() << "\n\n";
    ctx.t_io = secs(now() - ctx.t0);

    // Etapa 3: processar em chunks
    auto t1 = now();
    processar(m, pos_dados, DELIM, ctx);
    ctx.t_proc = secs(now() - t1);
    close_mmap(m);
    std::cout << "Total: " << ctx.total << " linhas em " << ctx.t_proc << " s\n";

    // Etapa 4: estatísticas
    auto t2 = now();
    calc_stats(ctx);
    ctx.t_stats = secs(now() - t2);

    // Relatório EDA
    std::cout << "\n============= EDA =============\n";
    for (auto& col : ctx.cols) std::cout << col.out.str();

    double total = secs(now() - ctx.t0);
    std::cout << "\n============= DESEMPENHO =============\n"
              << std::fixed << std::setprecision(3)
              << "Threads      : " << omp_get_max_threads() << "\n"
              << "Linhas       : " << ctx.total              << "\n"
              << "I/O + Init   : " << ctx.t_io               << " s\n"
              << "Processamento: " << ctx.t_proc             << " s\n"
              << "Estatisticas : " << ctx.t_stats            << " s\n"
              << "TOTAL        : " << total                  << " s\n"
              << "Vazao        : " << (long long)(ctx.total/total) << " linhas/s\n";
    return 0;
}
