import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Tactic

/-!
# CRISP / CLEAR — a scoped formal verification

Lean 4.19.0; mathlib c44e0c8ee63ca166450922a373c7409c5d26b00b.

The mathematical objects below use exact real arithmetic. They do not model
floating-point execution, FPGA arithmetic, the random-number generator, or
test-set accuracy. The certification composition theorems explicitly assume
the per-look error bounds: the PREDICT/rank-verification theorem is not silently
introduced as an axiom. CLEAR's loss theorem explicitly requires the exact
incoming derivative, so it does not apply to truncated or random feedback.
-/

namespace CrispClear

noncomputable section

/-! ## 1. The constant-input dendrite and exact retiming -/

def pole (r : ℝ) : ℝ := -Real.exp r

def decay (lam h : ℝ) : ℝ := Real.exp (lam * h)

def gain (lam h : ℝ) : ℝ := (Real.exp (lam * h) - 1) / lam

def flow (lam u h s : ℝ) : ℝ := decay lam h * s + gain lam h * u

theorem pole_neg (r : ℝ) : pole r < 0 := by
  exact neg_neg_of_pos (Real.exp_pos r)

theorem decay_stable (lam h : ℝ) (hlam : lam < 0) (hh : 0 < h) :
    0 < decay lam h ∧ decay lam h < 1 := by
  constructor
  · exact Real.exp_pos _
  · exact Real.exp_lt_one_iff.mpr (mul_neg_of_neg_of_pos hlam hh)

theorem flow_zero (lam u s : ℝ) : flow lam u 0 s = s := by
  simp [flow, decay, gain]

/-- Two intervals with the same held input compose exactly. -/
theorem flow_add (lam u h k s : ℝ) (hlam : lam ≠ 0) :
    flow lam u (h + k) s = flow lam u k (flow lam u h s) := by
  unfold flow decay gain
  rw [mul_add, Real.exp_add]
  field_simp
  ring

/-- The stated closed form satisfies ds/dt = lams + u. -/
theorem flow_solves_ode (lam u s t : ℝ) (hlam : lam ≠ 0) :
    HasDerivAt (fun t => flow lam u t s) (lam * flow lam u t s + u) t := by
  have he : HasDerivAt (fun t : ℝ => Real.exp (lam * t))
      (Real.exp (lam * t) * lam) t := by
    simpa using ((hasDerivAt_id t).const_mul lam).exp
  convert (he.mul_const s).add (((he.sub_const 1).div_const lam).mul_const u) using 1
  unfold flow decay gain
  field_simp
  ring

def heldRun (lam u h s : ℝ) : ℕ → ℝ
  | 0 => s
  | n + 1 => flow lam u h (heldRun lam u h s n)

theorem heldRun_eq_flow (lam u h s : ℝ) (hlam : lam ≠ 0) (n : ℕ) :
    heldRun lam u h s n = flow lam u ((n : ℝ) * h) s := by
  induction n with
  | zero => simp [heldRun, flow_zero]
  | succ n ih =>
    rw [heldRun, ih, ← flow_add lam u ((n : ℝ) * h) h s hlam]
    congr 2
    push_cast
    ring

/-- f fine steps reach precisely the coarse endpoint; f must be positive. -/
theorem retiming_exact (lam u h s : ℝ) (hlam : lam ≠ 0)
    (f : ℕ) (hf : 0 < f) :
    heldRun lam u (h / (f : ℝ)) s f = flow lam u h s := by
  rw [heldRun_eq_flow lam u (h / (f : ℝ)) s hlam f]
  have hf' : (f : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hf)
  congr 2
  field_simp

/-- The actual negative-exponential pole parameterization meets the premise. -/
theorem crisp_retiming_exact (r u h s : ℝ) (f : ℕ) (hf : 0 < f) :
    heldRun (pole r) u (h / (f : ℝ)) s f = flow (pole r) u h s :=
  retiming_exact _ _ _ _ (ne_of_lt (pole_neg r)) f hf

/-! ## 2. Eligibility derivatives, arbitrary finite trajectories, and an adjoint -/

def state (a b : ℝ → ℝ) (u : ℕ → ℝ → ℝ) (s₀ : ℝ → ℝ) : ℕ → ℝ → ℝ
  | 0 => s₀
  | n + 1 => fun q => a q * state a b u s₀ n q + b q * u n q

def eligibility (a b : ℝ → ℝ) (u : ℕ → ℝ → ℝ) (s₀ : ℝ → ℝ)
    (q da db ds₀ : ℝ) (du : ℕ → ℝ) : ℕ → ℝ
  | 0 => ds₀
  | n + 1 => a q * eligibility a b u s₀ q da db ds₀ du n
      + da * state a b u s₀ n q + db * u n q + b q * du n

/-- The eligibility recurrence is the genuine derivative of the state,
for every finite sequence length and every differentiable coefficient/input. -/
theorem eligibility_is_derivative
    (a b : ℝ → ℝ) (u : ℕ → ℝ → ℝ) (s₀ : ℝ → ℝ)
    (q da db ds₀ : ℝ) (du : ℕ → ℝ)
    (ha : HasDerivAt a da q) (hb : HasDerivAt b db q)
    (hs₀ : HasDerivAt s₀ ds₀ q) (hu : ∀ n, HasDerivAt (u n) (du n) q)
    (n : ℕ) :
    HasDerivAt (state a b u s₀ n) (eligibility a b u s₀ q da db ds₀ du n) q := by
  induction n with
  | zero => exact hs₀
  | succ n ih =>
    convert (ha.mul ih).add (hb.mul (hu n)) using 1
    simp only [eligibility]
    ring

/-- The same statement using mathlib's derivative operator. -/
theorem eligibility_eq_deriv
    (a b : ℝ → ℝ) (u : ℕ → ℝ → ℝ) (s₀ : ℝ → ℝ)
    (q da db ds₀ : ℝ) (du : ℕ → ℝ)
    (ha : HasDerivAt a da q) (hb : HasDerivAt b db q)
    (hs₀ : HasDerivAt s₀ ds₀ q) (hu : ∀ n, HasDerivAt (u n) (du n) q)
    (n : ℕ) :
    deriv (state a b u s₀ n) q = eligibility a b u s₀ q da db ds₀ du n :=
  (eligibility_is_derivative a b u s₀ q da db ds₀ du ha hb hs₀ hu n).deriv

def trajectory {K T : ℕ} (a b : Fin K → ℝ → ℝ)
    (u : Fin K → ℕ → ℝ → ℝ) (s₀ : Fin K → ℝ → ℝ)
    (q : ℝ) (i : Fin K × Fin T) : ℝ := state (a i.1) (b i.1) (u i.1) (s₀ i.1) i.2.val q

def tangent {K T : ℕ} (a b : Fin K → ℝ → ℝ)
    (u : Fin K → ℕ → ℝ → ℝ) (s₀ : Fin K → ℝ → ℝ)
    (q : ℝ) (da db ds₀ : Fin K → ℝ) (du : Fin K → ℕ → ℝ)
    (i : Fin K × Fin T) : ℝ :=
  eligibility (a i.1) (b i.1) (u i.1) (s₀ i.1) q (da i.1) (db i.1) (ds₀ i.1) (du i.1) i.2.val

theorem trajectory_has_derivative {K T : ℕ}
    (a b : Fin K → ℝ → ℝ) (u : Fin K → ℕ → ℝ → ℝ)
    (s₀ : Fin K → ℝ → ℝ) (q : ℝ) (da db ds₀ : Fin K → ℝ)
    (du : Fin K → ℕ → ℝ)
    (ha : ∀ k, HasDerivAt (a k) (da k) q)
    (hb : ∀ k, HasDerivAt (b k) (db k) q)
    (hs₀ : ∀ k, HasDerivAt (s₀ k) (ds₀ k) q)
    (hu : ∀ k n, HasDerivAt (u k n) (du k n) q) :
    HasDerivAt (trajectory (T := T) a b u s₀)
      (tangent a b u s₀ q da db ds₀ du) q := by
  apply hasDerivAt_pi.mpr
  intro i
  exact eligibility_is_derivative _ _ _ _ _ _ _ _ _ (ha i.1) (hb i.1) (hs₀ i.1) (hu i.1) i.2.val

/-- An arbitrary differentiable downstream function may couple all times and
poles. D is explicitly its exact incoming derivative. This does not prove that
a separately implemented normalization/soma has a claimed derivative. -/
theorem clear_loss_gradient {K T : ℕ}
    (a b : Fin K → ℝ → ℝ) (u : Fin K → ℕ → ℝ → ℝ)
    (s₀ : Fin K → ℝ → ℝ) (q : ℝ) (da db ds₀ : Fin K → ℝ)
    (du : Fin K → ℕ → ℝ)
    (ha : ∀ k, HasDerivAt (a k) (da k) q)
    (hb : ∀ k, HasDerivAt (b k) (db k) q)
    (hs₀ : ∀ k, HasDerivAt (s₀ k) (ds₀ k) q)
    (hu : ∀ k n, HasDerivAt (u k n) (du k n) q)
    (loss : (Fin K × Fin T → ℝ) → ℝ)
    (D : (Fin K × Fin T → ℝ) →L[ℝ] ℝ)
    (hD : HasFDerivAt loss D (trajectory a b u s₀ q)) :
    deriv (fun q => loss (trajectory a b u s₀ q)) q =
      D (tangent a b u s₀ q da db ds₀ du) := by
  exact (hD.comp_hasDerivAt q
    (trajectory_has_derivative a b u s₀ q da db ds₀ du ha hb hs₀ hu)).deriv

/-- Each list entry contains (local derivative injection, incoming learning
signal). The forward calculation propagates eligibility before using it. -/
def forwardCredit (a e : ℝ) : List (ℝ × ℝ) → ℝ
  | [] => 0
  | (d, w) :: xs => w * (a * e + d) + forwardCredit a (a * e + d) xs

/-- Reverse-time adjoint: first component is sensitivity to the initial state;
second component collects local parameter contributions. -/
def backwardCredit (a : ℝ) : List (ℝ × ℝ) → ℝ × ℝ
  | [] => (0, 0)
  | (d, w) :: xs =>
    let r := backwardCredit a xs
    (a * (w + r.1), d * (w + r.1) + r.2)

theorem eligibility_adjoint_identity (a e : ℝ) (xs : List (ℝ × ℝ)) :
    forwardCredit a e xs = (backwardCredit a xs).1 * e + (backwardCredit a xs).2 := by
  induction xs generalizing e with
  | nil => simp [forwardCredit, backwardCredit]
  | cons x xs ih =>
    rcases x with ⟨d, w⟩
    simp only [forwardCredit, backwardCredit, ih]
    ring

theorem zero_initial_eligibility_matches_adjoint (a : ℝ) (xs : List (ℝ × ℝ)) :
    forwardCredit a 0 xs = (backwardCredit a xs).2 := by
  simpa using eligibility_adjoint_identity a 0 xs

/-! ## 3. Explicit sequential selection and error-budget composition -/

open MeasureTheory
open scoped ENNReal

def firstSome {C : Type*} : List (Option C) → Option C
  | [] => none
  | none :: xs => firstSome xs
  | some c :: _ => some c

theorem firstSome_mem {C : Type*} (xs : List (Option C)) (c : C)
    (h : firstSome xs = some c) : some c ∈ xs := by
  induction xs with
  | nil => simp [firstSome] at h
  | cons x xs ih =>
    cases x with
    | none => exact List.mem_cons_of_mem none (ih h)
    | some d =>
      have hd : d = c := Option.some.inj h
      simp [hd]

def sequential {Ω C : Type*} {n : ℕ} (d : Fin n → Ω → Option C) (ω : Ω) : Option C :=
  firstSome (List.ofFn (fun i => d i ω))

theorem sequential_returns_a_look {Ω C : Type*} {n : ℕ}
    (d : Fin n → Ω → Option C) (ω : Ω) (c : C)
    (h : sequential d ω = some c) : ∃ i, d i ω = some c := by
  have hm := firstSome_mem (List.ofFn (fun i => d i ω)) c h
  simpa only [List.mem_ofFn] using hm

def wrong {Ω C : Type*} (d : Ω → Option C) (g : C) : Set Ω :=
  {ω | ∃ c, d ω = some c ∧ c ≠ g}

theorem sequential_wrong_subset {Ω C : Type*} {n : ℕ}
    (d : Fin n → Ω → Option C) (g : C) :
    wrong (sequential d) g ⊆ ⋃ i, wrong (d i) g := by
  intro ω hω
  rcases hω with ⟨c, hc, hcg⟩
  obtain ⟨i, hi⟩ := sequential_returns_a_look d ω c hc
  exact Set.mem_iUnion.mpr ⟨i, c, hi, hcg⟩

/-- No independence BETWEEN looks is assumed. Per-look bounds are hypotheses;
the same cumulative random draws may therefore be reused at successive looks. -/
theorem sequential_error_budget {Ω C : Type*} [MeasurableSpace Ω] {n : ℕ}
    (μ : Measure Ω) (d : Fin n → Ω → Option C) (g : C) (ε : Fin n → ℝ≥0∞)
    (hfixed : ∀ i, μ (wrong (d i) g) ≤ ε i) :
    μ (wrong (sequential d) g) ≤ ∑ i, ε i := by
  calc
    _ ≤ μ (⋃ i, wrong (d i) g) := measure_mono (sequential_wrong_subset d g)
    _ ≤ ∑ i, μ (wrong (d i) g) := measure_iUnion_fintype_le μ _
    _ ≤ ∑ i, ε i := Finset.sum_le_sum (fun i _ => hfixed i)

theorem five_looks_level_001 {Ω C : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (d : Fin 5 → Ω → Option C) (g : C)
    (hfixed : ∀ i, μ (wrong (d i) g) ≤ (1 / 5000 : ℝ≥0∞)) :
    μ (wrong (sequential d) g) ≤ (1 / 1000 : ℝ≥0∞) := by
  have h := sequential_error_budget μ d g (fun _ => (1 / 5000 : ℝ≥0∞)) hfixed
  convert h using 1
  simp only [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
  rw [← mul_div_assoc, mul_one]
  apply (ENNReal.div_eq_div_iff (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)).mpr
  norm_num

def disagreement {Ω C : Type*} (d r : Ω → Option C) : Set Ω :=
  {ω | ∃ c c', d ω = some c ∧ r ω = some c' ∧ c ≠ c'}

theorem disagreement_subset {Ω C : Type*} (d r : Ω → Option C) (g : C) :
    disagreement d r ⊆ wrong d g ∪ wrong r g := by
  intro ω hω
  rcases hω with ⟨c, c', hc, hc', hne⟩
  by_cases hcg : c = g
  · right
    refine ⟨c', hc', ?_⟩
    intro hc'g
    exact hne (hcg.trans hc'g.symm)
  · exact Or.inl ⟨c, hc, hcg⟩

/-- Unconditional disagreement bound. Independence of the reference is not
needed for this union step; its own error bound remains an explicit premise. -/
theorem reference_disagreement_budget {Ω C : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (d r : Ω → Option C) (g : C) (α αref : ℝ≥0∞)
    (hd : μ (wrong d g) ≤ α) (hr : μ (wrong r g) ≤ αref) :
    μ (disagreement d r) ≤ α + αref := by
  calc
    _ ≤ μ (wrong d g ∪ wrong r g) := measure_mono (disagreement_subset d r g)
    _ ≤ μ (wrong d g) + μ (wrong r g) := measure_union_le _ _
    _ ≤ α + αref := add_le_add hd hr

theorem reference_level_002 {Ω C : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (d r : Ω → Option C) (g : C)
    (hd : μ (wrong d g) ≤ (1 / 1000 : ℝ≥0∞))
    (hr : μ (wrong r g) ≤ (1 / 1000 : ℝ≥0∞)) :
    μ (disagreement d r) ≤ (2 / 1000 : ℝ≥0∞) := by
  have h := reference_disagreement_budget μ d r g _ _ hd hr
  simpa only [← ENNReal.add_div, one_add_one_eq_two] using h

/-- Factor-of-two correction for a symmetric binomial tail. This algebraic
lemma does not establish the multinomial rank-selection theorem. -/
theorem two_sided_cutoff (tail α : ℝ) (hα : α < 1) :
    min 1 (2 * tail) ≤ α ↔ tail ≤ α / 2 := by
  constructor
  · intro h
    rcases min_le_iff.mp h with h | h <;> linarith
  · intro h
    exact (min_le_right 1 (2 * tail)).trans (by linarith)

end

/-! ## 4. Exact integer boundary checks (no floating-point or native oracle) -/

def tailNumerator (m a : ℕ) : ℕ :=
  ((Finset.range (m + 1)).filter (fun k => a ≤ k)).sum (fun k => m.choose k)

/-- Acceptance at α=1/denom for top counts a≥b, computed with exact integers.
For positive denom≥2, the upper clipping of the p-value at 1 is immaterial. -/
def accepts (denom a b : ℕ) : Bool :=
  decide (b ≤ a ∧ 2 * denom * tailNumerator (a + b) a ≤ 2 ^ (a + b))

theorem fixed_unanimous_10_rejected : accepts 1000 10 0 = false := by decide
theorem fixed_unanimous_11_accepted : accepts 1000 11 0 = true := by decide
theorem sequential_unanimous_13_rejected : accepts 5000 13 0 = false := by decide
theorem sequential_unanimous_14_accepted : accepts 5000 14 0 = true := by decide
theorem first_look_16_0_accepted : accepts 5000 16 0 = true := by decide
theorem first_look_15_1_rejected : accepts 5000 15 1 = false := by decide
theorem fixed_look_15_1_accepted : accepts 1000 15 1 = true := by decide

end CrispClear

#print axioms CrispClear.pole_neg
#print axioms CrispClear.decay_stable
#print axioms CrispClear.flow_zero
#print axioms CrispClear.flow_add
#print axioms CrispClear.flow_solves_ode
#print axioms CrispClear.heldRun_eq_flow
#print axioms CrispClear.retiming_exact
#print axioms CrispClear.crisp_retiming_exact
#print axioms CrispClear.eligibility_is_derivative
#print axioms CrispClear.eligibility_eq_deriv
#print axioms CrispClear.trajectory_has_derivative
#print axioms CrispClear.clear_loss_gradient
#print axioms CrispClear.eligibility_adjoint_identity
#print axioms CrispClear.zero_initial_eligibility_matches_adjoint
#print axioms CrispClear.firstSome_mem
#print axioms CrispClear.sequential_returns_a_look
#print axioms CrispClear.sequential_wrong_subset
#print axioms CrispClear.sequential_error_budget
#print axioms CrispClear.five_looks_level_001
#print axioms CrispClear.disagreement_subset
#print axioms CrispClear.reference_disagreement_budget
#print axioms CrispClear.reference_level_002
#print axioms CrispClear.two_sided_cutoff
#print axioms CrispClear.fixed_unanimous_10_rejected
#print axioms CrispClear.fixed_unanimous_11_accepted
#print axioms CrispClear.sequential_unanimous_13_rejected
#print axioms CrispClear.sequential_unanimous_14_accepted
#print axioms CrispClear.first_look_16_0_accepted
#print axioms CrispClear.first_look_15_1_rejected
#print axioms CrispClear.fixed_look_15_1_accepted
