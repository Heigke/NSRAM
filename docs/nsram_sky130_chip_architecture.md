# NS-RAM Test Chip Architecture — SKY130 130nm CMOS

## Moonshot: First Open-Source Neuro-Synaptic RAM Silicon

**Target**: Fabricate and characterize NS-RAM cells on SKY130 130nm bulk CMOS,
exploiting the parasitic lateral NPN BJT in deep-N-well-isolated NMOS transistors
to demonstrate neuron spiking + synaptic charge trapping in standard CMOS.

**Based on**: Pazos et al., "Synaptic and neural behaviours in a standard
silicon transistor", Nature 640, 69-76 (2025). Zenodo: 13843362.

**Process**: SkyWater SKY130 (130nm, 5-metal, bulk CMOS with DNW, HV, BJT options)

---

## 1. Executive Summary

We exploit three key SKY130 features to build NS-RAM cells:

1. **Deep N-Well (DNW)** isolates the P-well from substrate → floating body
2. **Parasitic lateral NPN** (N+ source/body/drain) → avalanche + snapback
3. **5V/10V/20V HV devices** → sufficient voltage headroom for BVpar regime

The chip contains **7 cell variants** × **8 copies each** = 56 NS-RAM cells,
plus characterization structures, a 10-bit DAC, sense amplifiers, and digital
control via Wishbone bus. Total area: ~2 mm² of the 10.3 mm² user area.

---

## 2. The NS-RAM Cell on SKY130

### 2.1 Core Physics Mapping

| Pazos (SOI/FinFET) | SKY130 Implementation |
|--------------------|-----------------------|
| Floating bulk (SOI BOX isolates) | P-well inside DNW (DNW isolates from substrate) |
| Parasitic lateral NPN (S/B/D) | Same: N+ source = emitter, P-well = base, N+ drain = collector |
| Body capacitance Cb ≈ 1 pF | P-well junction cap + explicit MIM cap (sky130_fd_pr__cap_mim_m3_1) |
| Vg1 controls BVpar | Gate voltage modulates impact ionization rate |
| Vg2 controls neuron/synapse mode | Second transistor M2 sets body resistance (Rb) |
| Avalanche at BVpar = 3.5−1.5×Vg1 | SKY130 junction BV ~10.5-14V; HV devices up to 30V |
| SRH charge trapping in oxide | SONOS ONO stack (if available) or interface trap states |

### 2.2 Critical Difference: Bulk CMOS vs SOI

In SOI (Pazos original), the buried oxide perfectly isolates the body. Charge
retention is limited only by junction leakage and SRH recombination.

In SKY130 bulk CMOS with DNW, the P-well sits on top of the deep N-well.
The P-well/DNW junction is a reverse-biased diode that leaks. This means:
- **Body charge retention will be shorter** than SOI (maybe 1-100 ms vs seconds)
- **This IS the experiment** — measure actual retention in bulk CMOS
- If retention is too short for synapse mode, we add explicit MIM capacitors to
  increase Cb and extend tau

### 2.3 Voltage Budget

| Supply | Value | Use |
|--------|-------|-----|
| VDDA | 3.3V | Analog supply, adequate for 1.8V device avalanche |
| VCCD | 1.8V | Digital logic |
| V_EXT | 0-11V | External supply via io_analog pads for HV device characterization |

For the standard 1.8V NFET with DNW:
- Estimated parasitic BJT BVCEO: 6-9V
- Need external supply > 3.3V → use io_analog pad (no ESD, direct analog access)
- Or use HV devices (nfet_g5v0d10v5: 11V drain, nfet_20v0_iso: 22V drain)

---

## 3. Seven Cell Variants

Each variant tests a different aspect of NS-RAM physics. 8 copies of each
variant with staggered W/L for process variation characterization.

### CELL A: Standard 1.8V NFET + DNW (Primary NS-RAM Cell)

```
                     Vg1 (gate)
                       │
                  ┌────┤────┐
                  │  NFET   │    P-well floating inside DNW
    Source ───────┤  1v8    ├──────── Drain
    (Emitter)     │ W=10u  │        (Collector)
                  │ L=0.25u│
                  └────┬────┘
                       │
                  P-well body ← FLOATING (no body tie)
                       │
                  ┌────┴────┐
                  │   DNW   │ ← biased to VDDA or V_EXT
                  └─────────┘
```

- **Device**: `sky130_fd_pr__nfet_01v8` in isolated P-well (DNW underneath)
- **Body access**: Weak resistive contact (poly resistor, 10k-1M selectable) OR
  direct probe pad for measurement
- **W/L variants**: 10u/0.25u, 5u/0.5u, 20u/0.15u, 2u/1u (4 geometries × 2 copies)
- **Expected BVpar**: 6-9V (needs external supply through io_analog)
- **Tests**: I-V curves, avalanche onset, body charge retention, spiking frequency

### CELL B: 5V HV NFET + DNW

- **Device**: `sky130_fd_pr__nfet_g5v0d10v5` (5V gate oxide, 10.5V drain)
- **Advantage**: Thicker gate oxide → better charge retention, higher BVpar headroom
- **W/L**: 10u/0.5u, 5u/1u (2 geometries × 4 copies)
- **Expected BVpar**: 8-12V
- **Tests**: Same as Cell A but at higher voltages; compare retention time

### CELL C: ESD NFET (Pre-Characterized Snapback)

- **Device**: `sky130_fd_pr__esd_nfet_g5v0d10v5`
- **Advantage**: SkyWater characterizes this device FOR snapback behavior.
  Well-known trigger voltage, holding voltage, and It2 current.
- **Modification**: Leave body floating (remove standard body tie)
- **Tests**: Verify snapback → NS-RAM spiking equivalence

### CELL D: Native (Zero-VT) NFET + DNW

- **Device**: `sky130_fd_pr__nfet_03v3_nvt` or `nfet_05v0_nvt`
- **Advantage**: Zero threshold voltage → subthreshold floating body dynamics.
  The channel conducts at Vg=0, so body charging occurs at much lower voltages.
- **Tests**: Ultra-low-power NS-RAM operation, energy per spike

### CELL E: 20V Isolated NFET

- **Device**: `sky130_fd_pr__nfet_20v0_iso`
- **Advantage**: Built on DNW by design (no custom isolation needed).
  Drain voltage up to 22V → full Pazos voltage range accessible.
- **Tests**: Direct replication of Pazos I-V curves at original voltage scale

### CELL F: Explicit NPN BJT (Reference)

- **Device**: `sky130_fd_pr__npn_11v0` (poly-gated, beta~125)
- **Purpose**: Known-good BJT for calibration. Measure beta, BVCEO, avalanche
  current — compare to parasitic BJT in cells A-E.
- **Modification**: Add external capacitor (MIM) on base to emulate floating body

### CELL G: 2-Transistor NS-RAM Cell (Neuron + Synapse Control)

```
         Vg1                  Vg2
          │                    │
     ┌────┤────┐          ┌───┤───┐
     │  M1     │          │  M2   │
Src──┤  NFET   ├──Drain   │ PMOS  ├── Vbody_ctrl
     │ (neuron)│          │(mode) │
     └────┬────┘          └───┬───┘
          │                    │
          └────────┬───────────┘
              P-well body (floating)
                   │
              ┌────┴────┐
              │   DNW   │
              └─────────┘
```

- **M1**: `nfet_g5v0d10v5` (neuron transistor) — avalanche + spiking
- **M2**: `pfet_g5v0d10v5` (mode control) — Vg2 controls effective body resistance
  - Vg2=0V → M2 ON → low Rb → neuron mode (fast body discharge)
  - Vg2=3.3V → M2 OFF → high Rb → synapse mode (charge retention)
- **Cb**: MIM capacitor (1 pF, ~500 um²) on body node for explicit charge storage
- **This is the full NS-RAM cell** as described by Pazos et al.
- 8 copies with different MIM cap values (0.1, 0.5, 1.0, 5.0 pF)

---

## 4. On-Chip Periphery

### 4.1 Bias Generation — 10-bit R-2R DAC

- Generates precise Vg1 sweep (0-1.0V in 1mV steps) for I-V characterization
- Built from `sky130_fd_pr__res_high_po` (high-sheet poly resistors)
- Two independent DAC channels: one for Vg1, one for Vg2
- Output buffered by `sky130_fd_pr__pfet_01v8` source follower

### 4.2 Sense Amplifiers

- **Current sense**: Logarithmic transimpedance amp using `npn_05v5` BJT
  (Vout = Vt × ln(I_cell / I_ref)) — spans 6+ decades
- **Voltage sense**: Unity-gain buffer on body node for non-invasive readout
- **Spike detector**: Comparator (NMOS diff pair) with programmable threshold

### 4.3 Multiplexer + Analog I/O

- 8:1 analog MUX to route any cell's drain/body/gate to the 11 analog I/O pads
- Dedicated pads:
  - `io_analog[0]`: V_EXT (external high-voltage supply, 0-11V)
  - `io_analog[1]`: Vg1_ext (external gate bias override)
  - `io_analog[2]`: Vg2_ext (external mode control override)
  - `io_analog[3]`: I_drain (drain current sense output)
  - `io_analog[4]`: V_body (body voltage sense output)
  - `io_analog[5]`: V_spike (spike detector output)
  - `io_analog[6-10]`: Additional probe points / spare

### 4.4 Digital Control (Wishbone)

- Register map for:
  - Cell select (which of 56 cells is active)
  - DAC values (Vg1, Vg2)
  - MUX routing
  - Pulse generator (programmable width/frequency for transient characterization)
  - Spike counter (32-bit, counts spikes per cell over programmable window)
  - Temperature sensor readout (on-chip PNP V_BE)
- Interface: Wishbone bus from Caravel management SoC (accessible via SPI)

### 4.5 On-Chip Temperature Sensor

- PNP V_BE difference (sky130_fd_pr__pnp_05v5) at two current densities
- ~2 mV/K sensitivity
- Critical for measuring Tbv1 = -21.3e-6 /K temperature coefficient

---

## 5. Floorplan (10.3 mm² User Area)

```
┌──────────────────────────────────────────────────────┐
│                    CARAVEL HARNESS                    │
│  ┌────────────────────────────────────────────────┐  │
│  │              USER AREA (2.92 × 3.52 mm)        │  │
│  │                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │  CELL A  │  │  CELL B  │  │  CELL C  │     │  │
│  │  │  1.8V    │  │  5V HV   │  │  ESD     │     │  │
│  │  │  8 units │  │  8 units │  │  8 units │     │  │
│  │  │  100×200 │  │  100×200 │  │  100×200 │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘     │  │
│  │                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │  CELL D  │  │  CELL E  │  │  CELL F  │     │  │
│  │  │  Zero-VT │  │  20V iso │  │  NPN ref │     │  │
│  │  │  8 units │  │  8 units │  │  8 units │     │  │
│  │  │  100×200 │  │  100×200 │  │  100×200 │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘     │  │
│  │                                                │  │
│  │  ┌───────────────────────────────────────┐     │  │
│  │  │          CELL G: 2T NS-RAM ARRAY      │     │  │
│  │  │  8 full cells (M1 + M2 + MIM cap)     │     │  │
│  │  │  4 MIM cap values: 0.1, 0.5, 1, 5 pF │     │  │
│  │  │  ~200 × 600 um                        │     │  │
│  │  └───────────────────────────────────────┘     │  │
│  │                                                │  │
│  │  ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────────┐  │  │
│  │  │ 2× DAC  │ │ SENSE│ │ MUX  │ │ DIGITAL  │  │  │
│  │  │ 10-bit  │ │ AMPS │ │ 8:1  │ │ CONTROL  │  │  │
│  │  │ R-2R    │ │      │ │      │ │ Wishbone │  │  │
│  │  │ 150×300 │ │80×200│ │60×100│ │ 300×400  │  │  │
│  │  └─────────┘ └──────┘ └──────┘ └──────────┘  │  │
│  │                                                │  │
│  │  ┌───────────────────────────────────────┐     │  │
│  │  │       GUARD RINGS + DECOUPLING        │     │  │
│  │  │  (surrounds each cell block)          │     │  │
│  │  └───────────────────────────────────────┘     │  │
│  │                                                │  │
│  │  ┌───────────────────────────────────────┐     │  │
│  │  │    RESERVED FOR FUTURE: 8×8 ARRAY     │     │  │
│  │  │    (64 interconnected NS-RAM cells    │     │  │
│  │  │     for reservoir computing demo)     │     │  │
│  │  │    ~800 × 800 um                      │     │  │
│  │  └───────────────────────────────────────┘     │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  [38 GPIO] [11 Analog I/O] [Power Pads]              │
└──────────────────────────────────────────────────────┘
```

### Area Budget

| Block | Area (um²) | % of User Area |
|-------|-----------|---------------|
| Cell A-F (6 × 8 units) | 6 × 20,000 = 120,000 | 1.2% |
| Cell G (2T array, 8 units + MIM caps) | 120,000 | 1.2% |
| DAC (2 channels) | 45,000 | 0.4% |
| Sense amps + MUX | 22,000 | 0.2% |
| Digital control | 120,000 | 1.2% |
| Guard rings | 200,000 | 1.9% |
| Reserved 8×8 array | 640,000 | 6.2% |
| **Total used** | **~1,270,000** | **~12%** |
| **Remaining** | **~9,000,000** | **~88%** |

We use only ~12% of the user area. The remaining 88% can hold:
- Larger arrays (128+ NS-RAM cells for reservoir computing)
- Process control monitors
- Additional test structures

---

## 6. Key Experiments

### EXP 1: I-V Characterization (All Cells)

Sweep Vds from 0 to BVpar at 8+ Vg1 values (0, 0.1, 0.2, ..., 0.7V).
Measure drain current via external SMU through io_analog pads.

**Expected**: Exponential drain current rise near BVpar, with BVpar decreasing
as Vg1 increases (the Pazos signature).

**Key question**: Does BVpar follow 3.5 - 1.5×Vg1 on SKY130?

### EXP 2: Spiking Neuron (Cell A, B, G)

Apply constant Vds > BVpar, measure body voltage oscillations.
The floating body charges via avalanche → forward-biases body-source junction
→ body discharges (spike) → cycle repeats.

**Expected**: Self-sustained oscillation at 10 kHz - 1 MHz depending on Rb and Cb.

**Key question**: Does the parasitic BJT in bulk CMOS have enough gain for
self-oscillation? (SOI Bf ≈ 50, SKY130 npn_05v5 Bf ≈ 37 — close enough?)

### EXP 3: Synapse Mode (Cell G)

Switch M2 to high-Rb mode. Apply pulse train to drain. Measure body voltage
after 1, 10, 100, 1000 pulses. Wait and measure retention.

**Expected**: Body voltage increases with pulse count (potentiation) and decays
with time constant τ = Rb × Cb.

**Key question**: What is the actual charge retention time in bulk CMOS with DNW?
SOI achieves >10,000s. Bulk CMOS may be 1-100 ms. If too short, the explicit
MIM cap variants (Cell G with 5 pF) should extend to ~seconds.

### EXP 4: Temperature Dependence

Measure BVpar vs temperature using on-chip temperature sensor + external
thermal chuck. Verify Tbv1 = -21.3e-6 /K.

### EXP 5: Neuron-Synapse Mode Switching

Dynamically switch Vg2 between neuron and synapse modes while monitoring
body voltage. Verify that accumulated synaptic charge modulates subsequent
spiking threshold.

### EXP 6: Energy per Spike

Measure total energy per spike event: E = ∫ Vds × Ids dt over one spike.
Pazos reports 21 fJ total (4.7 fJ generation + 16.3 fJ integration).
SKY130 at 130nm with larger Cb may be 100-1000 fJ — still competitive.

### EXP 7: 8×8 Reservoir (if area permits)

64 interconnected NS-RAM cells with sparse recurrent connections via
on-chip resistive networks. Feed input through DAC, read population
spike rate. Attempt XOR benchmark.

---

## 7. Design Flow

### 7.1 Tools (All Open-Source)

```bash
# Install open_pdks + SKY130
git clone https://github.com/RTimothyEdwards/open_pdks
cd open_pdks && ./configure --enable-sky130-pdk && make && make install

# Schematic capture
sudo apt install xschem
# Get SKY130 symbols
git clone https://github.com/StefanSchippers/xschem_sky130

# SPICE simulation
sudo apt install ngspice

# Layout
sudo apt install magic

# LVS
sudo apt install netgen

# Digital flow (for Wishbone controller)
pip install openlane
```

### 7.2 Design Steps

1. **Schematic** (Xschem): Draw all 7 cell variants + periphery
2. **Simulate** (ngspice): Verify avalanche/spiking with SKY130 BSIM4 models
3. **Layout** (Magic): Custom analog cells with intentional floating body
4. **DRC** (Magic): Run DRC, waive body-tie violations with documentation
5. **LVS** (Netgen): Verify layout matches schematic
6. **Integrate** (OpenLane + Magic): Place digital controller, connect analog blocks
7. **GDS** (Magic): Generate final GDSII for submission
8. **Verify** (precheck): Run shuttle-specific precheck scripts

### 7.3 Simulation Strategy

Before tapeout, validate with ngspice using SKY130 models:

```spice
* NS-RAM Cell A: 1.8V NFET with floating body in DNW
.lib "./sky130A/libs.tech/ngspice/sky130.lib.spice" tt

* NFET with isolated P-well (body node accessible)
XM1 drain gate source body sky130_fd_pr__nfet_01v8 W=10u L=0.25u

* Body capacitance (explicit + parasitic)
C_body body gnd 1p

* Weak body resistance (controls neuron/synapse mode)
R_body body gnd 100k  ; adjustable: 10k (neuron) to 1M (synapse)

* DNW bias
V_dnw dnw_node gnd 3.3

* Bias
V_gate gate gnd 0.3
V_drain drain gnd DC 0 PULSE(0 5 0 1n 1n 10u 20u)
V_source source gnd 0

.tran 100n 100u
.probe V(body) I(V_drain)
.end
```

---

## 8. Fabrication Path (2026)

### Option A: Cadence/SkyWater Shuttle (Best for Full Chip)
- **Process**: SKY130 130nm
- **Cost**: ~$10,000-12,000
- **Timeline**: Next submission window TBD (check cadence.com/skywater)
- **Area**: Full 10.3 mm² user area in Caravel harness
- **Analog**: Full support (11 analog I/O pads)

### Option B: TinyTapeout IHP (Cheapest, BiCMOS)
- **Process**: IHP SG13G2 130nm BiCMOS
- **Cost**: $300/tile (or free via SwissChips for Swiss academic institutions)
- **Timeline**: ~6 months per shuttle
- **Advantage**: Real SiGe HBTs (not just parasitic BJTs)
- **Limitation**: Small tile area (~160×100 um), primarily digital
- **Best for**: Single NS-RAM cell proof-of-concept

### Option C: EUROPRACTICE Academic
- **Process**: XFAB XH018 (180nm **SOI** — true floating body!)
- **Cost**: ~$3,000-8,000/mm² (academic pricing)
- **Timeline**: 3-6 months
- **Advantage**: SOI process → real floating body, closest to Pazos original
- **Best for**: Definitive NS-RAM replication

### Option D: wafer.space (GF180MCU)
- **Process**: GlobalFoundries 180nm
- **Cost**: TBD (community-funded runs)
- **Timeline**: Quarterly runs planned
- **Status**: New platform, check wafer.space for updates

### Recommendation

**Immediate** ($300): Submit 1-2 NS-RAM cells on TinyTapeout IHP for proof of concept.

**Primary** ($10-12K): Full test chip on Cadence/SkyWater SKY130 with all 7 cell
variants, periphery, and 8×8 reservoir array.

**Dream** ($5-8K): XFAB XH018 SOI via EUROPRACTICE — true floating body physics,
definitive replication of Pazos et al.

---

## 9. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Parasitic BJT gain too low for self-oscillation | Medium | High | Include explicit NPN (Cell F) as backup; adjust Cb/Rb |
| Body charge retention too short (< 1 ms) | High | Medium | MIM cap variants (Cell G: 0.1-5 pF); increase to 10+ pF |
| Latchup from floating body | Medium | High | Guard rings, physical separation, separate supplies |
| BVpar too high for on-chip supply (> 3.3V) | High | Low | External supply via io_analog; HV device variants |
| DRC failures from missing body ties | Certain | Low | Document intentional design; waiver annotations |
| SKY130 SPICE models don't cover avalanche regime | Medium | Medium | Use ESD device models; characterize on silicon |
| Shuttle program closes/changes | Low | High | Design is portable to GF180/IHP; tools are open-source |

---

## 10. Bill of Materials (Test Equipment)

To characterize the fabricated chip:

| Equipment | Purpose | Cost | Source |
|-----------|---------|------|--------|
| Keithley 2400 SMU (or clone) | I-V sweeps, precise bias | $200-500 used | eBay |
| Analog Discovery 2/3 | Oscilloscope + function gen | $280-400 | Digilent |
| USB DAQ (NI/MCC) | Automated measurement | $150-300 | National Instruments |
| Thermal chuck or hot plate | Temperature characterization | $50-100 | Amazon |
| ZIF socket + breakout PCB | Chip mounting | $20-50 | JLCPCB + DigiKey |
| **Total test setup** | | **$700-1350** | |

---

## 11. Timeline

| Month | Milestone |
|-------|-----------|
| 0 | Schematic capture (Xschem) + ngspice simulation |
| 1 | Layout (Magic) + DRC/LVS clean |
| 2 | Integration with Caravel harness + precheck |
| 3 | Submit to shuttle |
| 6-9 | Chips arrive |
| 9-10 | Characterization + measurements |
| 11-12 | Paper: "NS-RAM in Standard Bulk CMOS: First Open-Source Silicon Demonstration" |

---

## 12. What We Prove

If even ONE cell variant demonstrates:
1. **Gate-controlled avalanche** (BVpar decreasing with Vg1) ✓
2. **Self-sustained body oscillation** (spiking at any frequency) ✓
3. **Body charge retention** (> 1 ms in synapse mode) ✓

Then we have **demonstrated NS-RAM in standard bulk CMOS** — the first open-source
replication of Pazos et al., proving the effect is not unique to SOI/FinFET.

If the 8×8 array works, we additionally demonstrate:
4. **Reservoir computing on NS-RAM silicon** (first ever)
5. **Energy per spike** measurement (target: < 1 pJ)

This would be a Nature-tier result: "Neuro-synaptic computing in standard CMOS."
