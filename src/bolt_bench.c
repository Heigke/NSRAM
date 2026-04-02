/*
 * BOLT Benchmark — 3 novel architectures on text8
 *
 * Compiles with: gcc -O3 -march=native -o bolt_bench bolt_bench.c -lm
 * Runs on CPU only (no GPU dependency).
 *
 * For GPU HW mechanisms: bolt_bench_gpu.hip adds HIP kernels that
 * use the 11 discovered GPU physics as computational primitives.
 *
 * Usage: ./bolt_bench <text8_path> [train_bytes] [val_bytes]
 */

#include "bolt_novel.h"
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════
 * SIMPLE BACKPROP-THROUGH-TIME (1 step, no unrolling)
 *
 * For online learning: we do 1-step BPTT after each byte.
 * This is the simplest possible training — no batching, no unrolling.
 * The architecture handles temporal credit via its internal recurrence.
 * ═══════════════════════════════════════════════════════════════════ */

/* Numerical gradient for Dense layers via finite differences */
/* (for correctness checking only — slow) */

/* Online SGD update for all Dense layers in a model */
typedef struct {
    Dense *layers;
    int n_layers;
} DenseList;

/* ═══════════════════════════════════════════════════════════════════
 * PCB TRAINING: prediction error as LOCAL learning signal
 *
 * Key insight: each layer's predictor learns to predict the layer above.
 * The prediction error IS the gradient — no global backprop needed!
 * This is biologically plausible and computationally cheap.
 * ═══════════════════════════════════════════════════════════════════ */

static void pcb_learn_step(PCB *m, uint8_t target, float *logits, float lr) {
    int D = m->d_model;
    float probs[256], grad_logits[256];

    /* Output layer: standard cross-entropy gradient */
    softmax_256(logits, probs);
    ce_grad(probs, target, grad_logits);

    /* Update output head */
    dense_accumulate_grad(&m->output_head, m->state[m->n_layers - 1], grad_logits);
    dense_update(&m->output_head, lr, 1e-5f);

    /* Per-layer LOCAL learning: minimize prediction error */
    for (int l = 0; l < m->n_layers - 1; l++) {
        /* Predictor[l] tried to predict state[l+1].
         * Error = state[l+1] - predicted[l].
         * Update predictor to reduce this error. */
        float *err = m->error[l];
        for (int i = 0; i < D; i++)
            err[i] = m->state[l + 1][i] - m->predicted[l][i];

        /* Gradient for predictor: dW += error ⊗ state[l] */
        dense_accumulate_grad(&m->predictor[l], m->state[l], err);
        dense_update(&m->predictor[l], lr * 0.5f, 1e-5f);

        /* Recurrent update: small global signal from output error */
        float *rec_grad = (float*)alloca(D * sizeof(float));
        float scale = 0.1f; /* Weaken global signal — local error dominates */
        for (int i = 0; i < D; i++)
            rec_grad[i] = err[i] * scale;

        float *concat = (float*)alloca(2 * D * sizeof(float));
        float *input = (l == 0) ? (m->embed + target * D) : m->error[l > 0 ? l - 1 : 0];
        memcpy(concat, input, D * sizeof(float));
        memcpy(concat + D, m->state[l], D * sizeof(float));
        dense_accumulate_grad(&m->recurrent[l], concat, rec_grad);
        dense_update(&m->recurrent[l], lr * 0.3f, 1e-5f);
    }

    /* Top layer: only gets output gradient */
    {
        int l = m->n_layers - 1;
        float *top_grad = (float*)alloca(D * sizeof(float));
        /* Backprop from output head through top state */
        for (int i = 0; i < D; i++) {
            float sum = 0;
            for (int j = 0; j < 256; j++)
                sum += m->output_head.W[j * D + i] * grad_logits[j];
            top_grad[i] = sum * (1.0f - m->state[l][i] * m->state[l][i]); /* tanh' */
        }
        float *concat = (float*)alloca(2 * D * sizeof(float));
        float *input = (l == 0) ? (m->embed + target * D) : m->error[l - 1];
        memcpy(concat, input, D * sizeof(float));
        memcpy(concat + D, m->state[l], D * sizeof(float));
        dense_accumulate_grad(&m->recurrent[l], concat, top_grad);
        dense_update(&m->recurrent[l], lr * 0.3f, 1e-5f);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * SRA TRAINING: only winning cells get updated
 *
 * Novel: losing cells' gradients are ZEROED → they freeze.
 * This is natural forgetting resistance — cells that specialized
 * on old data don't get corrupted by new data.
 * ═══════════════════════════════════════════════════════════════════ */

static void sra_learn_step(SRA *m, uint8_t target, float *logits,
                            int *active_cells, float lr) {
    int D = m->d_cell;
    float probs[256], grad_logits[256];

    softmax_256(logits, probs);
    ce_grad(probs, target, grad_logits);

    /* Only update WINNING cells */
    for (int k = 0; k < m->top_k; k++) {
        int c = active_cells[k];

        /* Update this cell's output head */
        dense_accumulate_grad(&m->local_head[c], m->state[c], grad_logits);
        dense_update(&m->local_head[c], lr, 1e-5f);

        /* Update this cell's recurrence (simplified: use output grad) */
        float *cell_grad = (float*)alloca(D * sizeof(float));
        for (int i = 0; i < D; i++) {
            float sum = 0;
            for (int j = 0; j < 256; j++)
                sum += m->local_head[c].W[j * D + i] * grad_logits[j];
            cell_grad[i] = sum * (1.0f - m->state[c][i] * m->state[c][i]);
        }
        float *concat = (float*)alloca(2 * D * sizeof(float));
        memcpy(concat, m->embed + target * D, D * sizeof(float));
        memcpy(concat + D, m->state[c], D * sizeof(float));
        dense_accumulate_grad(&m->recurrent[c], concat, cell_grad);
        dense_update(&m->recurrent[c], lr, 1e-5f);

        /* Update confidence scorer */
        /* Reward: confidence should be high when loss is low */
        float loss = cross_entropy(probs, target);
        float conf_target = (loss < 2.0f) ? 1.0f : -1.0f;
        float conf_val;
        dense_forward(&m->confidence[c], m->state[c], &conf_val);
        float conf_grad = sigmoid(conf_val) - ((conf_target > 0) ? 1.0f : 0.0f);
        dense_accumulate_grad(&m->confidence[c], m->state[c], &conf_grad);
        dense_update(&m->confidence[c], lr * 0.1f, 1e-5f);
    }
    /* LOSING cells: NO UPDATE → frozen → forgetting resistance */
}

/* ═══════════════════════════════════════════════════════════════════
 * LBN TRAINING: τ modulates gradient flow
 *
 * Novel: neurons with high τ (slow) get SMALLER gradients.
 * This naturally protects consolidated knowledge.
 * τ itself is learned — the model learns WHAT to protect.
 * ═══════════════════════════════════════════════════════════════════ */

static void lbn_learn_step(LBN *m, uint8_t target, float *logits, float lr) {
    int D = m->d_model;
    float probs[256], grad_logits[256];

    softmax_256(logits, probs);
    ce_grad(probs, target, grad_logits);

    /* Update output head */
    dense_accumulate_grad(&m->output_head, m->state[m->n_layers - 1], grad_logits);
    dense_update(&m->output_head, lr, 1e-5f);

    /* Backprop through layers (simplified 1-step) */
    float *grad = (float*)alloca(D * sizeof(float));
    for (int i = 0; i < D; i++) {
        float sum = 0;
        for (int j = 0; j < 256; j++)
            sum += m->output_head.W[j * D + i] * grad_logits[j];
        grad[i] = sum;
    }

    for (int l = m->n_layers - 1; l >= 0; l--) {
        /* τ-modulated gradient: high τ → small update */
        for (int i = 0; i < D; i++) {
            float tau = m->tau_current[l][i];
            grad[i] *= (1.0f / tau); /* KEY: τ gates gradient magnitude */
        }

        /* Update W_in and W_rec */
        float *inp = (l == 0) ? (m->embed + target * D) : m->state[l - 1];
        dense_accumulate_grad(&m->W_in[l], inp, grad);
        dense_update(&m->W_in[l], lr, 1e-5f);

        dense_accumulate_grad(&m->W_rec[l], m->state[l], grad);
        dense_update(&m->W_rec[l], lr, 1e-5f);

        /* Update W_tau (learns WHAT to protect) */
        float *tau_grad = (float*)alloca(D * sizeof(float));
        float *concat = (float*)alloca(2 * D * sizeof(float));
        memcpy(concat, inp, D * sizeof(float));
        memcpy(concat + D, m->state[l], D * sizeof(float));
        for (int i = 0; i < D; i++)
            tau_grad[i] = grad[i] * (-m->state[l][i]) / (m->tau_current[l][i] * m->tau_current[l][i])
                          * (m->tau_max - m->tau_min) * sigmoid(0) * (1 - sigmoid(0)); /* approximate */
        dense_accumulate_grad(&m->W_tau[l], concat, tau_grad);
        dense_update(&m->W_tau[l], lr * 0.3f, 1e-5f);

        /* Propagate gradient to layer below (simplified) */
        if (l > 0) {
            float *new_grad = (float*)alloca(D * sizeof(float));
            for (int i = 0; i < D; i++) {
                float sum = 0;
                for (int j = 0; j < D; j++)
                    sum += m->W_in[l].W[j * D + i] * grad[j]; /* approximate */
                new_grad[i] = sum;
            }
            memcpy(grad, new_grad, D * sizeof(float));
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * SIMPLE RNN BASELINE (for comparison)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int d_model;
    float *state;
    Dense recurrent;  /* [embed + state] → state */
    Dense output;     /* state → 256 */
    float *embed;
} SimpleRNN;

static SimpleRNN rnn_new(int d_model) {
    SimpleRNN m;
    m.d_model = d_model;
    m.state = (float*)calloc(d_model, sizeof(float));
    m.embed = (float*)calloc(256 * d_model, sizeof(float));
    for (int i = 0; i < 256 * d_model; i++) m.embed[i] = xavier(256, d_model);
    m.recurrent = dense_new(d_model + d_model, d_model);
    m.output = dense_new(d_model, 256);
    return m;
}

static void rnn_forward(SimpleRNN *m, uint8_t byte_in, float *logits) {
    int D = m->d_model;
    float *concat = (float*)alloca(2 * D * sizeof(float));
    float *raw = (float*)alloca(D * sizeof(float));
    memcpy(concat, m->embed + byte_in * D, D * sizeof(float));
    memcpy(concat + D, m->state, D * sizeof(float));
    dense_forward(&m->recurrent, concat, raw);
    for (int i = 0; i < D; i++) m->state[i] = fast_tanh(raw[i]);
    dense_forward(&m->output, m->state, logits);
}

static void rnn_learn(SimpleRNN *m, uint8_t target, float *logits, float lr) {
    int D = m->d_model;
    float probs[256], grad_logits[256];
    softmax_256(logits, probs);
    ce_grad(probs, target, grad_logits);
    dense_accumulate_grad(&m->output, m->state, grad_logits);
    dense_update(&m->output, lr, 1e-5f);

    float *grad = (float*)alloca(D * sizeof(float));
    for (int i = 0; i < D; i++) {
        float sum = 0;
        for (int j = 0; j < 256; j++)
            sum += m->output.W[j * D + i] * grad_logits[j];
        grad[i] = sum * (1.0f - m->state[i] * m->state[i]);
    }
    float *concat = (float*)alloca(2 * D * sizeof(float));
    memcpy(concat, m->embed + target * D, D * sizeof(float));
    memcpy(concat + D, m->state, D * sizeof(float));
    dense_accumulate_grad(&m->recurrent, concat, grad);
    dense_update(&m->recurrent, lr, 1e-5f);
}

static int rnn_params(const SimpleRNN *m) {
    int D = m->d_model;
    return 256 * D + (2*D)*D + D + D*256 + 256;
}

static void rnn_free(SimpleRNN *m) {
    free(m->state); free(m->embed);
    dense_free(&m->recurrent); dense_free(&m->output);
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN BENCHMARK
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <text8_path> [train_bytes] [val_bytes]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int train_n = argc > 2 ? atoi(argv[2]) : 2000000;
    int val_n = argc > 3 ? atoi(argv[3]) : 200000;

    /* Load data */
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    int total = train_n + val_n;
    uint8_t *data = (uint8_t*)malloc(total);
    int read_n = fread(data, 1, total, f);
    fclose(f);
    if (read_n < total) { fprintf(stderr, "Only read %d bytes\n", read_n); total = read_n; }

    uint8_t *train = data;
    uint8_t *val = data + train_n;
    int actual_val = total - train_n;
    if (actual_val < 0) actual_val = 0;

    printf("BOLT Novel Architectures — text8 Benchmark\n");
    printf("Train: %d bytes, Val: %d bytes\n\n", train_n, actual_val);

    srand(42);
    int D = 96;   /* Bigger for better results */
    float lr = 0.003f;

    /* Create models */
    PCB pcb = pcb_new(D, 4);
    SRA sra = sra_new(8, D, 3);   /* 8 cells, top-3 active */
    LBN lbn = lbn_new(D, 3, 1.0f, 20.0f);
    SimpleRNN rnn = rnn_new(D);

    printf("Model sizes:\n");
    printf("  PCB  (Predictive Coding):  %d params\n", pcb_params(&pcb));
    printf("  SRA  (Self-Routing):       %d params\n", sra_params(&sra));
    printf("  LBN  (Liquid Byte):        %d params\n", lbn_params(&lbn));
    printf("  RNN  (baseline):           %d params\n", rnn_params(&rnn));
    printf("\n");

    /* ═══ TRAINING (online, byte by byte) ═══ */
    float logits[256];
    float probs[256];
    int active_cells[SRA_MAX_CELLS];

    double pcb_loss = 0, sra_loss = 0, lbn_loss = 0, rnn_loss = 0;
    int pcb_correct = 0, sra_correct = 0, lbn_correct = 0, rnn_correct = 0;
    int report_every = 100000;

    printf("Training (online, byte-by-byte)...\n");
    printf("  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "Bytes", "PCB_bpc", "PCB_acc", "SRA_bpc", "SRA_acc",
           "LBN_bpc", "LBN_acc", "RNN_bpc", "RNN_acc");

    clock_t t0 = clock();

    for (int i = 0; i < train_n - 1; i++) {
        uint8_t x = train[i];
        uint8_t y = train[i + 1];

        /* PCB */
        pcb_forward(&pcb, x, logits);
        softmax_256(logits, probs);
        pcb_loss += cross_entropy(probs, y);
        if (logits[y] >= logits[0]) { /* quick argmax check */
            int best = 0;
            for (int j = 1; j < 256; j++) if (logits[j] > logits[best]) best = j;
            if (best == y) pcb_correct++;
        }
        pcb_learn_step(&pcb, y, logits, lr);

        /* SRA */
        sra_forward(&sra, x, logits, active_cells);
        softmax_256(logits, probs);
        sra_loss += cross_entropy(probs, y);
        { int best = 0;
          for (int j = 1; j < 256; j++) if (logits[j] > logits[best]) best = j;
          if (best == y) sra_correct++; }
        sra_learn_step(&sra, y, logits, active_cells, lr);

        /* LBN */
        lbn_forward(&lbn, x, logits);
        softmax_256(logits, probs);
        lbn_loss += cross_entropy(probs, y);
        { int best = 0;
          for (int j = 1; j < 256; j++) if (logits[j] > logits[best]) best = j;
          if (best == y) lbn_correct++; }
        lbn_learn_step(&lbn, y, logits, lr);

        /* RNN baseline */
        rnn_forward(&rnn, x, logits);
        softmax_256(logits, probs);
        rnn_loss += cross_entropy(probs, y);
        { int best = 0;
          for (int j = 1; j < 256; j++) if (logits[j] > logits[best]) best = j;
          if (best == y) rnn_correct++; }
        rnn_learn(&rnn, y, logits, lr);

        if ((i + 1) % report_every == 0) {
            double n = report_every;
            printf("  %8d  %7.3f  %7.1f%%  %7.3f  %7.1f%%  %7.3f  %7.1f%%  %7.3f  %7.1f%%\n",
                   i + 1,
                   pcb_loss / n / log(2), pcb_correct / n * 100,
                   sra_loss / n / log(2), sra_correct / n * 100,
                   lbn_loss / n / log(2), lbn_correct / n * 100,
                   rnn_loss / n / log(2), rnn_correct / n * 100);
            pcb_loss = sra_loss = lbn_loss = rnn_loss = 0;
            pcb_correct = sra_correct = lbn_correct = rnn_correct = 0;
        }
    }

    clock_t t1 = clock();
    double train_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("\nTraining: %.1f sec (%.0f bytes/sec)\n", train_time, train_n / train_time);

    /* ═══ VALIDATION ═══ */
    printf("\nValidation (%d bytes)...\n", actual_val);
    pcb_loss = sra_loss = lbn_loss = rnn_loss = 0;
    pcb_correct = sra_correct = lbn_correct = rnn_correct = 0;

    /* Reset states */
    for (int l = 0; l < pcb.n_layers; l++) memset(pcb.state[l], 0, D * sizeof(float));
    for (int c = 0; c < sra.n_cells; c++) memset(sra.state[c], 0, D * sizeof(float));
    for (int l = 0; l < lbn.n_layers; l++) memset(lbn.state[l], 0, D * sizeof(float));
    memset(rnn.state, 0, D * sizeof(float));

    for (int i = 0; i < actual_val - 1; i++) {
        uint8_t x = val[i], y = val[i + 1];

        pcb_forward(&pcb, x, logits);
        softmax_256(logits, probs);
        pcb_loss += cross_entropy(probs, y);
        { int best=0; for(int j=1;j<256;j++) if(logits[j]>logits[best]) best=j;
          if(best==y) pcb_correct++; }

        sra_forward(&sra, x, logits, NULL);
        softmax_256(logits, probs);
        sra_loss += cross_entropy(probs, y);
        { int best=0; for(int j=1;j<256;j++) if(logits[j]>logits[best]) best=j;
          if(best==y) sra_correct++; }

        lbn_forward(&lbn, x, logits);
        softmax_256(logits, probs);
        lbn_loss += cross_entropy(probs, y);
        { int best=0; for(int j=1;j<256;j++) if(logits[j]>logits[best]) best=j;
          if(best==y) lbn_correct++; }

        rnn_forward(&rnn, x, logits);
        softmax_256(logits, probs);
        rnn_loss += cross_entropy(probs, y);
        { int best=0; for(int j=1;j<256;j++) if(logits[j]>logits[best]) best=j;
          if(best==y) rnn_correct++; }
    }

    double vn = actual_val - 1;
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  FINAL RESULTS — text8 validation\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  %-25s %8s %8s %10s\n", "Model", "BPC", "Acc", "Params");
    printf("  ─────────────────────────────────────────────────────\n");
    printf("  %-25s %7.3f %7.1f%% %10d\n", "PCB (Predictive Coding)", pcb_loss/vn/log(2), pcb_correct/vn*100, pcb_params(&pcb));
    printf("  %-25s %7.3f %7.1f%% %10d\n", "SRA (Self-Routing)",     sra_loss/vn/log(2), sra_correct/vn*100, sra_params(&sra));
    printf("  %-25s %7.3f %7.1f%% %10d\n", "LBN (Liquid Byte)",      lbn_loss/vn/log(2), lbn_correct/vn*100, lbn_params(&lbn));
    printf("  %-25s %7.3f %7.1f%% %10d\n", "RNN (baseline)",         rnn_loss/vn/log(2), rnn_correct/vn*100, rnn_params(&rnn));
    printf("═══════════════════════════════════════════════════════════\n");

    /* SRA routing stats */
    printf("\nSRA cell activation counts:\n");
    for (int c = 0; c < sra.n_cells; c++)
        printf("  Cell %d: %d wins (%.1f%%)\n", c, sra.cell_wins[c],
               100.0 * sra.cell_wins[c] / (sra.total_steps > 0 ? sra.total_steps : 1));

    /* LBN tau distribution */
    printf("\nLBN avg tau per layer:\n");
    for (int l = 0; l < lbn.n_layers; l++) {
        float sum = 0;
        for (int i = 0; i < D; i++) sum += lbn.tau_current[l][i];
        printf("  Layer %d: avg_tau = %.2f\n", l, sum / D);
    }

    /* Literature comparison */
    printf("\nLiterature comparison (text8 BPC):\n");
    printf("  SOTA:         1.038 (277M params, TXL+dyn_eval)\n");
    printf("  Small LSTM:   1.59  (3.3M params)\n");
    printf("  Our models:   ~%dK params (no published baseline at this scale)\n",
           pcb_params(&pcb) / 1000);

    /* Cleanup */
    pcb_free(&pcb); sra_free(&sra); lbn_free(&lbn); rnn_free(&rnn);
    free(data);
    return 0;
}
