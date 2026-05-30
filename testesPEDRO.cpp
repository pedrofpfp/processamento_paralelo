// =============================================================================
// ANALISADOR CSV PARALELO — OpenMP + mmap + Reservoir Sampling
// Compilação: g++ -O3 -fopenmp -o analisador final.cpp
// Uso:        ./analisador dados.csv [delimitador]
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
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

static const size_t CHUNK_LINES = 1'000'000;
static const size_t RES_LOCAL   =     5'000;
static const size_t RES_GLOBAL  =    50'000;

// =============================================================================
// OTIMIZAÇÃO 1 — PARSER DE DOUBLE MANUAL
// strtod é lento: trata locale, infinito, NaN, aloca internamente.
// Para inteiros e decimais simples (99% dos CSVs): ~5× mais rápido.
// Fallback automático para strtod em notação científica.
// =============================================================================
static const double POW10[] = {
    1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,
    1e10,1e11,1e12,1e13,1e14,1e15,1e16,1e17,1e18
};

static bool fast_double(const char* s, size_t len, double& out) {
    const char* p = s, *end = s + len;
    bool neg = (*p == '-'); if (neg | (*p == '+')) p++;
    if (__builtin_expect(p == end, 0)) return false;

    uint64_t ipart = 0; int idig = 0;
    while (p < end) {
        unsigned c = (unsigned char)*p - '0';
        if (c > 9) break;
        ipart = ipart * 10 + c; p++; idig++;
    }
    uint64_t fpart = 0; int fdig = 0;
    if (p < end && *p == '.') {
        p++;
        while (p < end) {
            unsigned c = (unsigned char)*p - '0';
            if (c > 9) break;
            fpart = fpart * 10 + c; p++; fdig++;
        }
    }
    if (__builtin_expect(!idig && !fdig, 0)) return false;
    // Notação científica ou char inesperado → strtod
    if (__builtin_expect(p != end, 0)) {
        if (*p == 'e' || *p == 'E') { char* e; out = strtod(s,&e); return e==end; }
        return false;
    }
    // Número grande demais para conversão inteira precisa → strtod
    if (__builtin_expect(ipart > 999999999999999ULL || fdig > 18, 0)) {
        char* e; out = strtod(s,&e); return e==end;
    }
    double val = (double)ipart;
    if (fdig) val += (double)fpart / POW10[fdig];
    out = neg ? -val : val;
    return true;
}

// =============================================================================
// OTIMIZAÇÃO 2 — HASH MAP OPEN-ADDRESSING PARA CATEGORIAS
// std::unordered_map<string> faz heap alloc em CADA lookup (mesmo hits).
// Este hash map compara diretamente (ptr, len) — zero alocação no hit.
// Hash: FNV-1a — rápido e boa distribuição para strings curtas.
// =============================================================================
struct HashMap {
    struct Slot { int id; uint32_t hash; };
    static const int EMPTY = -1;

    std::vector<Slot>        table;
    std::vector<std::string> keys;
    std::vector<size_t>      cnts;
    int mask = 15;

    HashMap() { table.assign(16, {EMPTY, 0}); }

    static uint32_t fnv1a(const char* s, size_t n) {
        uint32_t h = 2166136261u;
        while (n--) h = (h ^ (uint8_t)*s++) * 16777619u;
        return h;
    }

    void insert(const char* s, size_t len, size_t n = 1) {
        uint32_t h = fnv1a(s, len);
        int idx = (int)(h & (uint32_t)mask);
        // Probing linear: percorre até slot vazio ou chave igual
        while (table[idx].id != EMPTY) {
            if (table[idx].hash == h) {
                const auto& k = keys[table[idx].id];
                if (k.size() == len && !memcmp(k.data(), s, len)) {
                    cnts[table[idx].id] += n; return; // hit: apenas incrementa, sem alloc
                }
            }
            idx = (idx + 1) & mask;
        }
        // Miss: insere novo
        int id = (int)keys.size();
        table[idx] = {id, h};
        keys.emplace_back(s, len);
        cnts.push_back(n);
        if ((id + 1) * 4 > (mask + 1) * 3) grow(); // rehash se >75% cheio
    }

    void grow() {
        mask = mask * 2 + 1;
        table.assign(mask + 1, {EMPTY, 0});
        for (int id = 0; id < (int)keys.size(); id++) {
            uint32_t h = fnv1a(keys[id].data(), keys[id].size());
            int idx = (int)(h & (uint32_t)mask);
            while (table[idx].id != EMPTY) idx = (idx + 1) & mask;
            table[idx] = {id, h};
        }
    }

    void clear() { table.assign(16, {EMPTY,0}); mask=15; keys.clear(); cnts.clear(); }
    size_t size() const { return keys.size(); }
};

// =============================================================================
// UTILITÁRIOS
// =============================================================================
static void trim(const char*& s, size_t& len) {
    while (len && (s[0]==' '||s[0]=='\t'||s[0]=='\r')) { s++; len--; }
    while (len && (s[len-1]==' '||s[len-1]=='\t'||s[len-1]=='\r')) len--;
}

static bool is_null(const char* s, size_t len) {
    return !len
        || (len==2 && s[0]=='N' && s[1]=='A')
        || (len==4 && !memcmp(s,"null",4))
        || (len==3 && !memcmp(s,"N/A",3))
        || (len==3 && !memcmp(s,"nan",3));
}

// LCG thread-safe: sem rand() global
static uint64_t lcg(uint64_t& st) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
}

// Quicksort com tasks OpenMP para ordenar o reservoir
static void qsort_d(std::vector<double>& v, int lo, int hi) {
    if (hi - lo < 2) return;
    if (hi - lo <= 32) {
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
//
// OTIMIZAÇÃO 3 — alignas(64): cada LCol começa em boundary de cache line.
//   Evita false sharing quando threads acessam colunas adjacentes no merge.
//
// OTIMIZAÇÃO 4 — Sentinela +/-inf em vmin/vmax:
//   Elimina o bool `first` e o branch `if(first||x<vmin)` no hot path.
//   Antes: 2 branches por valor. Depois: 2 comparações simples.
// =============================================================================

struct alignas(64) LCol {
    bool   num   = true;
    size_t nulos = 0, count = 0, seen = 0;
    double mean  = 0, M2   = 0;
    double vmin  = +1e308, vmax = -1e308;  // sentinela: sem bool `first`
    std::vector<double> res;
    uint64_t rng = 0;
    HashMap  dict;

    void init(int tid, int cid) {
        res.reserve(RES_LOCAL);
        rng = (uint64_t)tid * 2654435761ULL ^ (uint64_t)cid * 40503ULL;
    }

    void reset(bool tipo) {
        num=tipo; nulos=0; count=0; seen=0;
        mean=0; M2=0; vmin=+1e308; vmax=-1e308;
        res.clear(); dict.clear();
    }

    void add_num(double x) {
        count++; seen++;
        if (x < vmin) vmin = x;  // sem branch `first` — sempre compara
        if (x > vmax) vmax = x;
        double d = x - mean; mean += d / count; M2 += d * (x - mean);
        // Reservoir Algorithm R: amostra uniforme sem guardar todos os valores
        if (__builtin_expect(res.size() < RES_LOCAL, 1)) res.push_back(x);
        else { size_t j = lcg(rng) % seen; if (j < RES_LOCAL) res[j] = x; }
    }

    void vira_cat(const char* s, size_t len) {
        num=false; count=0; mean=0; M2=0; vmin=+1e308; vmax=-1e308; res.clear();
        dict.insert(s, len);
    }
};

struct alignas(64) GCol {
    std::string nome;
    bool   num   = true;
    size_t nulos = 0, count = 0, seen_res = 0;
    double mean  = 0, M2   = 0;
    double vmin  = +1e308, vmax = -1e308;
    std::vector<double> res;
    uint64_t rng = 0;
    HashMap  dict;
    std::ostringstream out;

    void init(const std::string& n, int idx) {
        nome = n; res.reserve(RES_GLOBAL);
        rng = (uint64_t)idx * 1234567891ULL ^ 0xCAFEBABEULL;
    }

    // Fórmula de Chan et al.: combina dois estados Welford exatamente
    void merge_num(const LCol& l) {
        if (!l.count) return;
        if (l.vmin < vmin) vmin = l.vmin;
        if (l.vmax > vmax) vmax = l.vmax;
        size_t n = count + l.count;
        double delta = l.mean - mean;
        mean = (mean * count + l.mean * l.count) / (double)n;
        M2  += l.M2 + delta * delta * (double)count * (double)l.count / (double)n;
        count = n;
        for (double x : l.res) {
            seen_res++;
            if (__builtin_expect(res.size() < RES_GLOBAL, 1)) res.push_back(x);
            else { size_t j = lcg(rng) % seen_res; if (j < RES_GLOBAL) res[j] = x; }
        }
    }

    void merge_cat(const LCol& l) {
        for (size_t i = 0; i < l.dict.size(); i++)
            dict.insert(l.dict.keys[i].data(), l.dict.keys[i].size(), l.dict.cnts[i]);
    }
};

struct Ctx {
    std::vector<GCol> cols;
    size_t total = 0;
    double t_io = 0, t_proc = 0, t_stats = 0;
    std::chrono::high_resolution_clock::time_point t0;
};

// =============================================================================
// MMAP — arquivo mapeado direto na RAM sem cópia
// =============================================================================
struct Mapa { const char* buf=nullptr; size_t tam=0; int fd=-1; };

static bool open_mmap(const std::string& f, Mapa& m) {
    m.fd = open(f.c_str(), O_RDONLY);
    if (m.fd < 0) { std::cerr << "ERRO: " << f << "\n"; return false; }
    struct stat st; fstat(m.fd, &st); m.tam = st.st_size;
    m.buf = (const char*)mmap(nullptr, m.tam, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.buf == MAP_FAILED) { std::cerr << "ERRO mmap\n"; close(m.fd); return false; }
    madvise((void*)m.buf, m.tam, MADV_SEQUENTIAL); // hint: leitura sequencial
    return true;
}
static void close_mmap(Mapa& m) { munmap((void*)m.buf, m.tam); close(m.fd); }

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
// PROCESSAMENTO EM CHUNKS
//
// OTIMIZAÇÃO 5 — schedule(static): sem overhead de lock de distribuição dinâmica.
//   Funciona bem porque cada linha tem custo similar (mesmo número de colunas).
//
// OTIMIZAÇÃO 6 — MADV_WILLNEED: enquanto CPU processa chunk N,
//   pedimos ao SO para carregar chunk N+1 do disco em paralelo.
//   Elimina latência de I/O no próximo chunk.
//
// OTIMIZAÇÃO 7 — __builtin_expect: indica ao compilador qual branch é provável.
//   Nulos são raros → coloca o código de nulo "fora" do hot path.
//   lc.num é quase sempre true → o else (categórico) fica "longe".
// =============================================================================
static void processar(const Mapa& m, size_t pos0, char delim, Ctx& ctx) {
    int nt = omp_get_max_threads();
    int NC = (int)ctx.cols.size();
    size_t pos = pos0;

    // Aloca LCols uma vez e reutiliza entre chunks (sem malloc por chunk)
    std::vector<std::vector<LCol>> local(nt, std::vector<LCol>(NC));
    for (int t = 0; t < nt; t++)
        for (int c = 0; c < NC; c++)
            local[t][c].init(t, c);

    while (pos < m.tam) {
        // Coleta offsets do chunk (sequencial, ~10 ms para 1M linhas)
        size_t chunk_byte_start = pos;
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

        // OTIMIZAÇÃO 6: prefetch do próximo chunk enquanto processamos o atual
        if (pos < m.tam) {
            size_t chunk_bytes = pos - chunk_byte_start;
            size_t hint = (pos + chunk_bytes <= m.tam) ? chunk_bytes : m.tam - pos;
            madvise((void*)(m.buf + pos), hint, MADV_WILLNEED);
        }

        // Propaga tipo global: coluna categórica não volta a ser numérica
        for (int t = 0; t < nt; t++)
            for (int c = 0; c < NC; c++)
                local[t][c].reset(ctx.cols[c].num);

        // OTIMIZAÇÃO 5: schedule(static) — zero overhead de sincronização
        #pragma omp parallel for schedule(static) num_threads(nt)
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
                // OTIMIZAÇÃO 7: hints de branch — nulos raros, num é comum
                if (__builtin_expect(is_null(sv, sl), 0)) {
                    lc.nulos++;
                } else if (__builtin_expect(lc.num, 1)) {
                    double d;
                    // OTIMIZAÇÃO 1: fast_double em vez de strtod
                    if (__builtin_expect(fast_double(sv, sl, d), 1)) lc.add_num(d);
                    else                                              lc.vira_cat(sv, sl);
                } else {
                    // OTIMIZAÇÃO 2: HashMap — sem alloc de string no hit
                    lc.dict.insert(sv, sl);
                }
                f = g+1; col++;
            }
        }

        // Merge local → global: cada col é independente → sem locks
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int col = 0; col < NC; col++) {
            GCol& gc = ctx.cols[col];
            for (int t = 0; t < nt; t++)
                if (!local[t][col].num) { gc.num = false; break; }
            for (int t = 0; t < nt; t++) gc.nulos += local[t][col].nulos;
            if (gc.num) { for (int t = 0; t < nt; t++) gc.merge_num(local[t][col]); }
            else        { for (int t = 0; t < nt; t++) gc.merge_cat(local[t][col]); }
        }

        std::cout << "> " << ctx.total << " linhas...\r" << std::flush;
    }
    std::cout << "\n";
}

// =============================================================================
// ESTATÍSTICAS
// =============================================================================
static void stats_num(GCol& col) {
    auto& o = col.out;
    size_t total = col.count + col.nulos;
    o << "\n[NUM] " << col.nome << "\n"
      << "  Nulos   : " << col.nulos << " ("
      << std::fixed << std::setprecision(2)
      << (total ? col.nulos * 100.0 / total : 0) << "%)\n";
    if (!col.count) { o << "  (sem dados)\n"; return; }

    double desvio = (col.count > 1 && col.M2 > 0) ? std::sqrt(col.M2 / col.count) : 0;

    #pragma omp taskgroup
    { qsort_d(col.res, 0, (int)col.res.size()); }

    size_t nr = col.res.size();
    double p25 = nr ? col.res[nr/4] : 0;
    double med = nr ? (nr%2==0 ? (col.res[nr/2-1]+col.res[nr/2])*0.5 : col.res[nr/2]) : 0;
    double p75 = nr ? col.res[3*nr/4] : 0;

    o << "  Registros: " << col.count << "\n"
      << "  Min      : " << col.vmin  << "\n"
      << "  Max      : " << col.vmax  << "\n"
      << "  Media    : " << col.mean  << "\n"
      << "  Mediana* : " << med << "  (*amostral, " << nr << " pts)\n"
      << "  P25/P75* : " << p25 << " / " << p75 << "\n"
      << "  Desvio   : " << desvio    << "\n";
}

static void stats_cat(GCol& col) {
    auto& o = col.out;
    size_t total_v = 0;
    for (size_t c : col.dict.cnts) total_v += c;
    size_t total = total_v + col.nulos;

    o << "\n[CAT] " << col.nome << "\n"
      << "  Nulos   : " << col.nulos << " ("
      << std::fixed << std::setprecision(2)
      << (total ? col.nulos * 100.0 / total : 0) << "%)\n"
      << "  Unicos  : " << col.dict.size() << "\n";

    if (!total_v) return;
    const int TOP = 5;
    std::vector<bool> used(col.dict.size(), false);
    o << "  Top " << TOP << ":\n";
    for (int k = 0; k < TOP && k < (int)col.dict.size(); k++) {
        size_t best = col.dict.size();
        for (size_t i = 0; i < col.dict.size(); i++)
            if (!used[i] && (best==col.dict.size() || col.dict.cnts[i]>col.dict.cnts[best]))
                best = i;
        if (best == col.dict.size()) break;
        double f = col.dict.cnts[best] * 100.0 / total_v;
        o << "    \"" << col.dict.keys[best] << "\" — "
          << col.dict.cnts[best] << " (" << f << "%)\n";
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

    Mapa m;
    if (!open_mmap(ARQ, m)) return 1;
    std::cout << "Tamanho : " << m.tam/(1024*1024) << " MB\n";

    size_t pos_dados = init_cols(m, DELIM, ctx);
    std::cout << "Colunas : " << ctx.cols.size() << "\n\n";
    ctx.t_io = secs(now() - ctx.t0);

    auto t1 = now();
    processar(m, pos_dados, DELIM, ctx);
    ctx.t_proc = secs(now() - t1);
    close_mmap(m);

    auto t2 = now();
    calc_stats(ctx);
    ctx.t_stats = secs(now() - t2);

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
