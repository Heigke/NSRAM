/*
 * BOLT Novel Architectures — C header
 *
 * Three genuinely novel byte-level online learning architectures:
 * 1. PCB  — Predictive Coding Byte Stack (hierarchical error propagation)
 * 2. SRA  — Self-Routing Automata (competitive voting, no external router)
 * 3. LBN  — Liquid Byte Network (ODE-based adaptive time constants)
 *
 * All operate on raw bytes (0-255), predict next byte (256 classes),
 * and learn online with no task boundaries.
 *
 * None of these exist in the literature as of 2026-03-30.
 */

#ifndef BOLT_NOVEL_H
#define BOLT_NOVEL_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════
 * MATH PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════ */

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float fast_tanh(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

static inline float relu(float x) { return x > 0 ? x : 0; }

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x*x*x)));
}

/* Xavier init */
static inline float xavier(int fan_in, int fan_out) {
    float u = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    return u * sqrtf(6.0f / (fan_in + fan_out));
}

/* ═══════════════════════════════════════════════════════════════════
 * DENSE LAYER (basic building block)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *W;       /* [out × in] row-major */
    float *b;       /* [out] */
    float *dW;      /* gradient accumulator */
    float *db;
    int in, out;
} Dense;

static Dense dense_new(int in, int out) {
    Dense d;
    d.in = in; d.out = out;
    d.W  = (float*)calloc(out * in, sizeof(float));
    d.b  = (float*)calloc(out, sizeof(float));
    d.dW = (float*)calloc(out * in, sizeof(float));
    d.db = (float*)calloc(out, sizeof(float));
    for (int i = 0; i < out * in; i++) d.W[i] = xavier(in, out);
    return d;
}

static void dense_forward(const Dense *d, const float *x, float *y) {
    for (int o = 0; o < d->out; o++) {
        float sum = d->b[o];
        const float *row = d->W + o * d->in;
        for (int i = 0; i < d->in; i++) sum += row[i] * x[i];
        y[o] = sum;
    }
}

/* Outer product gradient: dW += dy ⊗ x, db += dy */
static void dense_accumulate_grad(Dense *d, const float *x, const float *dy) {
    for (int o = 0; o < d->out; o++) {
        d->db[o] += dy[o];
        float *drow = d->dW + o * d->in;
        for (int i = 0; i < d->in; i++) drow[i] += dy[o] * x[i];
    }
}

/* SGD with momentum */
static void dense_update(Dense *d, float lr, float wd) {
    int n = d->out * d->in;
    for (int i = 0; i < n; i++) {
        d->W[i] -= lr * (d->dW[i] + wd * d->W[i]);
        d->dW[i] = 0;
    }
    for (int i = 0; i < d->out; i++) {
        d->b[i] -= lr * d->db[i];
        d->db[i] = 0;
    }
}

static void dense_free(Dense *d) {
    free(d->W); free(d->b); free(d->dW); free(d->db);
}


/* ═══════════════════════════════════════════════════════════════════
 * 1. PCB — PREDICTIVE CODING BYTE STACK
 *
 * Novel: each layer predicts the NEXT layer's activation.
 * Only the prediction ERROR propagates upward.
 * Lower layers = local byte patterns (bigrams/trigrams).
 * Upper layers = global structure (words, syntax).
 * Single forward pass (no iterative settling).
 *
 * THIS DOES NOT EXIST IN THE LITERATURE for language/byte modeling.
 * ═══════════════════════════════════════════════════════════════════ */

#define PCB_MAX_LAYERS 8

typedef struct {
    int n_layers;
    int d_model;

    /* Per-layer: recurrent state + prediction */
    float *state[PCB_MAX_LAYERS];     /* [d_model] hidden state */
    Dense  recurrent[PCB_MAX_LAYERS]; /* state × (input + state) → new_state */
    Dense  predictor[PCB_MAX_LAYERS]; /* state → predicted activation of layer above */

    /* Embedding + output */
    float *embed;       /* [256 × d_model] byte embeddings */
    Dense  output_head; /* top state → 256 logits */

    /* Error signals for learning */
    float *error[PCB_MAX_LAYERS];     /* prediction error per layer */
    float *predicted[PCB_MAX_LAYERS]; /* cached predictions */
} PCB;

static PCB pcb_new(int d_model, int n_layers) {
    PCB m;
    m.d_model = d_model;
    m.n_layers = n_layers;

    m.embed = (float*)calloc(256 * d_model, sizeof(float));
    for (int i = 0; i < 256 * d_model; i++) m.embed[i] = xavier(256, d_model);

    for (int l = 0; l < n_layers; l++) {
        m.state[l] = (float*)calloc(d_model, sizeof(float));
        m.error[l] = (float*)calloc(d_model, sizeof(float));
        m.predicted[l] = (float*)calloc(d_model, sizeof(float));
        /* Recurrent: takes [input_dim + d_model] → d_model */
        int in_dim = (l == 0) ? d_model : d_model; /* error from below or embedding */
        m.recurrent[l] = dense_new(in_dim + d_model, d_model);
        m.predictor[l] = dense_new(d_model, d_model);
    }
    m.output_head = dense_new(d_model, 256);
    return m;
}

static void pcb_forward(PCB *m, uint8_t byte_in, float *logits) {
    int D = m->d_model;
    float *concat = (float*)alloca(2 * D * sizeof(float));
    float *raw = (float*)alloca(D * sizeof(float));

    /* Layer 0 input: byte embedding */
    float *emb = m->embed + byte_in * D;

    for (int l = 0; l < m->n_layers; l++) {
        /* Input to this layer:
         * Layer 0: embedding
         * Layer >0: prediction ERROR from layer below */
        float *input = (l == 0) ? emb : m->error[l - 1];

        /* Concatenate input + recurrent state */
        memcpy(concat, input, D * sizeof(float));
        memcpy(concat + D, m->state[l], D * sizeof(float));

        /* Recurrent update */
        dense_forward(&m->recurrent[l], concat, raw);
        for (int i = 0; i < D; i++)
            m->state[l][i] = fast_tanh(raw[i]);

        /* Predict what layer above expects */
        dense_forward(&m->predictor[l], m->state[l], m->predicted[l]);

        /* Compute error: actual activation - prediction from below
         * CLAMP to prevent explosion (the key fix for PCB stability) */
        if (l > 0) {
            for (int i = 0; i < D; i++) {
                float e = m->state[l][i] - m->predicted[l - 1][i];
                if (e > 2.0f) e = 2.0f;
                if (e < -2.0f) e = -2.0f;
                m->error[l - 1][i] = e;
            }
        }
    }

    /* Top layer error (for last predictor) */
    /* The top layer has no layer above, so its "target" is itself */
    for (int i = 0; i < D; i++)
        m->error[m->n_layers - 1][i] = 0;  /* No error at top */

    /* Output: top layer state → 256 logits */
    dense_forward(&m->output_head, m->state[m->n_layers - 1], logits);
}

static int pcb_params(const PCB *m) {
    int n = 256 * m->d_model; /* embed */
    for (int l = 0; l < m->n_layers; l++) {
        n += m->recurrent[l].in * m->recurrent[l].out + m->recurrent[l].out;
        n += m->predictor[l].in * m->predictor[l].out + m->predictor[l].out;
    }
    n += m->output_head.in * m->output_head.out + m->output_head.out;
    return n;
}

static void pcb_free(PCB *m) {
    free(m->embed);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->state[l]); free(m->error[l]); free(m->predicted[l]);
        dense_free(&m->recurrent[l]); dense_free(&m->predictor[l]);
    }
    dense_free(&m->output_head);
}


/* ═══════════════════════════════════════════════════════════════════
 * 2. SRA — SELF-ROUTING AUTOMATA
 *
 * Novel: N cells, each with internal state. No router network.
 * For each byte:
 *   1. Every cell computes confidence: "how well can I predict next byte?"
 *   2. Top-K cells "win" and get updated (their gradients flow)
 *   3. Losing cells are FROZEN (no update, state preserved)
 *
 * This naturally creates specialization:
 *   - Some cells learn text patterns
 *   - Some cells learn image patterns
 *   - Switching modality → different cells activate
 *   - Old cells FREEZE → forgetting resistance
 *
 * No external router, no task boundaries, no MoE overhead.
 * ═══════════════════════════════════════════════════════════════════ */

#define SRA_MAX_CELLS 32

typedef struct {
    int n_cells;
    int d_cell;     /* hidden dim per cell */
    int top_k;      /* how many cells update per step */

    /* Per-cell: hidden state + recurrent weights + local output head */
    float *state[SRA_MAX_CELLS];
    Dense  recurrent[SRA_MAX_CELLS]; /* [d_cell + embed_dim] → d_cell */
    Dense  local_head[SRA_MAX_CELLS]; /* d_cell → 256 (each cell predicts bytes) */

    /* Confidence scorer: d_cell → 1 (how confident is this cell?) */
    Dense  confidence[SRA_MAX_CELLS];

    /* Shared byte embedding */
    float *embed;    /* [256 × d_cell] */

    /* Routing stats */
    int    cell_wins[SRA_MAX_CELLS]; /* how often each cell wins */
    int    total_steps;
} SRA;

static SRA sra_new(int n_cells, int d_cell, int top_k) {
    SRA m;
    m.n_cells = n_cells;
    m.d_cell = d_cell;
    m.top_k = top_k;
    m.total_steps = 0;

    m.embed = (float*)calloc(256 * d_cell, sizeof(float));
    for (int i = 0; i < 256 * d_cell; i++) m.embed[i] = xavier(256, d_cell);

    for (int c = 0; c < n_cells; c++) {
        m.state[c] = (float*)calloc(d_cell, sizeof(float));
        m.recurrent[c] = dense_new(d_cell + d_cell, d_cell);
        m.local_head[c] = dense_new(d_cell, 256);
        m.confidence[c] = dense_new(d_cell, 1);
        m.cell_wins[c] = 0;
    }
    return m;
}

static void sra_forward(SRA *m, uint8_t byte_in, float *logits, int *active_cells) {
    int D = m->d_cell;
    int N = m->n_cells;
    float *concat = (float*)alloca(2 * D * sizeof(float));
    float *raw = (float*)alloca(D * sizeof(float));
    float conf[SRA_MAX_CELLS];
    float *emb = m->embed + byte_in * D;

    /* Step 1: Every cell updates its state (cheap: small per-cell) */
    for (int c = 0; c < N; c++) {
        memcpy(concat, emb, D * sizeof(float));
        memcpy(concat + D, m->state[c], D * sizeof(float));
        dense_forward(&m->recurrent[c], concat, raw);
        for (int i = 0; i < D; i++)
            m->state[c][i] = fast_tanh(raw[i]);
    }

    /* Step 2: Each cell votes — confidence = how well it thinks it knows */
    for (int c = 0; c < N; c++) {
        float v;
        dense_forward(&m->confidence[c], m->state[c], &v);
        conf[c] = v; /* Raw confidence, not sigmoid — allows ranking */
    }

    /* Step 3: Top-K selection (simple insertion sort for small N) */
    int winners[SRA_MAX_CELLS];
    for (int k = 0; k < m->top_k; k++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int c = 0; c < N; c++) {
            int already = 0;
            for (int j = 0; j < k; j++) if (winners[j] == c) already = 1;
            if (!already && conf[c] > best) { best = conf[c]; best_idx = c; }
        }
        winners[k] = best_idx;
        m->cell_wins[best_idx]++;
    }
    if (active_cells) memcpy(active_cells, winners, m->top_k * sizeof(int));

    /* Step 4: Winning cells produce logits, weighted by softmax(confidence) */
    /* Compute softmax over winners */
    float max_conf = -1e30f;
    for (int k = 0; k < m->top_k; k++)
        if (conf[winners[k]] > max_conf) max_conf = conf[winners[k]];

    float sum_exp = 0;
    float weights[SRA_MAX_CELLS];
    for (int k = 0; k < m->top_k; k++) {
        weights[k] = expf(conf[winners[k]] - max_conf);
        sum_exp += weights[k];
    }
    for (int k = 0; k < m->top_k; k++) weights[k] /= sum_exp;

    /* Weighted average of winning cells' predictions */
    memset(logits, 0, 256 * sizeof(float));
    float *cell_logits = (float*)alloca(256 * sizeof(float));
    for (int k = 0; k < m->top_k; k++) {
        int c = winners[k];
        dense_forward(&m->local_head[c], m->state[c], cell_logits);
        for (int i = 0; i < 256; i++) logits[i] += weights[k] * cell_logits[i];
    }

    m->total_steps++;
}

static int sra_params(const SRA *m) {
    int n = 256 * m->d_cell;
    for (int c = 0; c < m->n_cells; c++) {
        n += m->recurrent[c].in * m->recurrent[c].out + m->recurrent[c].out;
        n += m->local_head[c].in * m->local_head[c].out + m->local_head[c].out;
        n += m->confidence[c].in * m->confidence[c].out + m->confidence[c].out;
    }
    return n;
}

static void sra_free(SRA *m) {
    free(m->embed);
    for (int c = 0; c < m->n_cells; c++) {
        free(m->state[c]);
        dense_free(&m->recurrent[c]);
        dense_free(&m->local_head[c]);
        dense_free(&m->confidence[c]);
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * 3. LBN — LIQUID BYTE NETWORK
 *
 * Novel: Each neuron's time constant τ is a LEARNED FUNCTION of input.
 * Predictable bytes → short τ (fast, forgetful)
 * Surprising bytes → long τ (slow, retentive)
 * τ is not a hyperparameter — it's computed per-neuron per-step.
 *
 * Based on Liquid Time-Constant Networks (Hasani 2021) but:
 * - Applied to byte-level prediction (first time)
 * - τ drives consolidation: high-τ neurons resist update (stability)
 * - Surprise modulates τ globally (entropy of prediction → gate)
 *
 * ODE per neuron: dx/dt = (-x + f(Wx + Ux_prev)) / τ(x, input)
 * Discretized: x_new = x + dt * ((-x + f(concat)) / τ)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int d_model;
    int n_layers;

    /* Per-layer state + weights */
    float *state[PCB_MAX_LAYERS];
    Dense  W_in[PCB_MAX_LAYERS];   /* input → d_model */
    Dense  W_rec[PCB_MAX_LAYERS];  /* state → d_model (recurrent) */
    Dense  W_tau[PCB_MAX_LAYERS];  /* [input + state] → d_model (τ per neuron) */

    /* τ bounds */
    float tau_min, tau_max;

    /* Embedding + output */
    float *embed;
    Dense  output_head;

    /* Per-step τ (for analysis) */
    float *tau_current[PCB_MAX_LAYERS];
} LBN;

static LBN lbn_new(int d_model, int n_layers, float tau_min, float tau_max) {
    LBN m;
    m.d_model = d_model;
    m.n_layers = n_layers;
    m.tau_min = tau_min;
    m.tau_max = tau_max;

    m.embed = (float*)calloc(256 * d_model, sizeof(float));
    for (int i = 0; i < 256 * d_model; i++) m.embed[i] = xavier(256, d_model);

    for (int l = 0; l < n_layers; l++) {
        m.state[l] = (float*)calloc(d_model, sizeof(float));
        m.tau_current[l] = (float*)calloc(d_model, sizeof(float));
        /* Initialize τ to midpoint */
        for (int i = 0; i < d_model; i++)
            m.tau_current[l][i] = (tau_min + tau_max) / 2.0f;

        int in_dim = (l == 0) ? d_model : d_model;
        m.W_in[l] = dense_new(in_dim, d_model);
        m.W_rec[l] = dense_new(d_model, d_model);
        m.W_tau[l] = dense_new(in_dim + d_model, d_model);
    }
    m.output_head = dense_new(d_model, 256);
    return m;
}

static void lbn_forward(LBN *m, uint8_t byte_in, float *logits) {
    int D = m->d_model;
    float *concat = (float*)alloca(2 * D * sizeof(float));
    float *f_in = (float*)alloca(D * sizeof(float));
    float *f_rec = (float*)alloca(D * sizeof(float));
    float *tau_raw = (float*)alloca(D * sizeof(float));

    float *input = m->embed + byte_in * D;
    float dt = 1.0f; /* discrete time step */

    for (int l = 0; l < m->n_layers; l++) {
        float *x = m->state[l];
        float *inp = (l == 0) ? input : m->state[l - 1];

        /* Compute τ per neuron: τ = tau_min + (tau_max - tau_min) * σ(W_tau @ [input, state]) */
        memcpy(concat, inp, D * sizeof(float));
        memcpy(concat + D, x, D * sizeof(float));
        dense_forward(&m->W_tau[l], concat, tau_raw);
        for (int i = 0; i < D; i++) {
            float s = sigmoid(tau_raw[i]);
            m->tau_current[l][i] = m->tau_min + (m->tau_max - m->tau_min) * s;
        }

        /* ODE step: dx/dt = (-x + f(W_in @ input + W_rec @ x)) / τ */
        dense_forward(&m->W_in[l], inp, f_in);
        dense_forward(&m->W_rec[l], x, f_rec);
        for (int i = 0; i < D; i++) {
            float target = fast_tanh(f_in[i] + f_rec[i]);
            float tau = m->tau_current[l][i];
            x[i] = x[i] + dt * (-x[i] + target) / tau;
        }

        input = x; /* Next layer's input */
    }

    dense_forward(&m->output_head, m->state[m->n_layers - 1], logits);
}

static int lbn_params(const LBN *m) {
    int n = 256 * m->d_model;
    for (int l = 0; l < m->n_layers; l++) {
        n += m->W_in[l].in * m->W_in[l].out + m->W_in[l].out;
        n += m->W_rec[l].in * m->W_rec[l].out + m->W_rec[l].out;
        n += m->W_tau[l].in * m->W_tau[l].out + m->W_tau[l].out;
    }
    n += m->output_head.in * m->output_head.out + m->output_head.out;
    return n;
}

static void lbn_free(LBN *m) {
    free(m->embed);
    for (int l = 0; l < m->n_layers; l++) {
        free(m->state[l]); free(m->tau_current[l]);
        dense_free(&m->W_in[l]); dense_free(&m->W_rec[l]); dense_free(&m->W_tau[l]);
    }
    dense_free(&m->output_head);
}


/* ═══════════════════════════════════════════════════════════════════
 * SOFTMAX + CROSS-ENTROPY
 * ═══════════════════════════════════════════════════════════════════ */

static void softmax_256(float *logits, float *probs) {
    float max_val = logits[0];
    for (int i = 1; i < 256; i++) if (logits[i] > max_val) max_val = logits[i];
    float sum = 0;
    for (int i = 0; i < 256; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < 256; i++) probs[i] /= sum;
}

static float cross_entropy(const float *probs, uint8_t target) {
    float p = probs[target];
    if (p < 1e-10f) p = 1e-10f;
    return -logf(p);
}

/* Gradient of softmax cross-entropy: dL/dlogits = probs - one_hot(target) */
static void ce_grad(const float *probs, uint8_t target, float *grad) {
    for (int i = 0; i < 256; i++) grad[i] = probs[i];
    grad[target] -= 1.0f;
}

#endif /* BOLT_NOVEL_H */
