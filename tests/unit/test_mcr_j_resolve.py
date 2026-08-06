# tests/unit/test_mcr_j_resolve.py
"""
Testes para a infraestrutura MCR-ALS usada pelo J-Resolve.
Valida mcr_als() com misturas sintéticas: separação de componentes,
proporcionalidade de concentração, e robustez a n_samples=1.
"""
import numpy as np
import pytest
from gui.utils.mcr_utils import mcr_als, simulate_lorentzian_profile


# ---------------------------------------------------------------------------
# Helpers para construir espectros sintéticos
# ---------------------------------------------------------------------------

def _lorentzian(ppm, centre, hwhm, amp=1.0):
    return amp * hwhm**2 / ((ppm - centre)**2 + hwhm**2)


def _build_mixture(ppm, components, concentrations, noise_std=0.0, rng=None):
    """
    Constrói matriz X (n_pts × n_samples) com misturas sintéticas.

    components    : lista de arrays (n_pts,) — perfis espectrais puros
    concentrations: lista de arrays (n_samples,) — concentrações por amostra
    """
    n_pts     = len(ppm)
    n_samples = len(concentrations[0])
    X = np.zeros((n_pts, n_samples))
    for comp, conc in zip(components, concentrations):
        X += np.outer(comp, conc)
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        X += rng.normal(0, noise_std, X.shape)
    return np.clip(X, 0, None)


# ---------------------------------------------------------------------------
# Testes: mcr_als básico
# ---------------------------------------------------------------------------

class TestMCRALSBasic:

    def setup_method(self):
        self.ppm  = np.linspace(2.5, 1.5, 256)
        self.rng  = np.random.default_rng(0)

        # Dois metabolitos com picos Lorentzianos separados
        self.S_A = _lorentzian(self.ppm, 2.2, 0.005)   # pico em 2.2 ppm
        self.S_B = _lorentzian(self.ppm, 1.8, 0.005)   # pico em 1.8 ppm
        # Normalizar
        self.S_A /= self.S_A.max() + 1e-12
        self.S_B /= self.S_B.max() + 1e-12

    def test_two_components_separated(self):
        """MCR deve separar dois perfis Lorentzianos bem distintos."""
        n_samples = 10
        C_A = self.rng.uniform(0.5, 2.0, n_samples)
        C_B = self.rng.uniform(0.5, 2.0, n_samples)
        X = _build_mixture(self.ppm, [self.S_A, self.S_B], [C_A, C_B])

        S_init = np.column_stack([self.S_A, self.S_B])
        C, S, metrics = mcr_als(X, n_components=2, S_init=S_init, max_iter=300, tol=1e-7)

        assert C.shape == (n_samples, 2)
        assert S.shape == (len(self.ppm), 2)
        # LOF deve ser baixa para mistura limpa
        assert metrics['lof'] < 10.0, f"LOF={metrics['lof']:.1f}% demasiado elevada"

    def test_concentration_proportional(self):
        """Com concentrações anti-correlacionadas, MCR deve preservar a razão média."""
        # Concentrações variáveis mas A sempre ≈2× B — MCR precisa de variação
        # entre amostras para decompor componentes correctamente
        n_samples = 20
        t = np.linspace(0, 2 * np.pi, n_samples)
        C_A = 2.0 + 0.5 * np.sin(t)        # varia em torno de 2.0
        C_B = 1.0 + 0.5 * np.cos(t)        # varia em torno de 1.0 (anti-correlado)
        C_A = np.clip(C_A, 0.1, None)
        C_B = np.clip(C_B, 0.1, None)
        X = _build_mixture(self.ppm, [self.S_A, self.S_B], [C_A, C_B])

        S_init = np.column_stack([self.S_A, self.S_B])
        C, S, metrics = mcr_als(X, n_components=2, S_init=S_init, max_iter=500, tol=1e-8)

        # Identificar coluna que corresponde a A (apex em 2.2 ppm)
        apex_idx = np.argmin(np.abs(self.ppm - 2.2))
        col_a = int(np.argmax(S[apex_idx, :]))
        col_b = 1 - col_a

        ratio = C[:, col_a].mean() / (C[:, col_b].mean() + 1e-12)
        # Razão média deve ser próxima de 2.0 (tolerância larga por ambiguidade de escala MCR)
        assert 1.2 < ratio < 3.5, (
            f"Razão de concentração MCR={ratio:.2f}, esperado entre 1.2 e 3.5")

    def test_lof_below_5pct_clean_mixture(self):
        """LOF < 5% para mistura sintética sem ruído."""
        n_samples = 8
        C_A = self.rng.uniform(1.0, 3.0, n_samples)
        C_B = self.rng.uniform(1.0, 3.0, n_samples)
        X = _build_mixture(self.ppm, [self.S_A, self.S_B], [C_A, C_B], noise_std=0.0)

        S_init = np.column_stack([self.S_A, self.S_B])
        C, S, metrics = mcr_als(X, n_components=2, S_init=S_init, max_iter=500, tol=1e-8)

        assert metrics['lof'] < 5.0, f"LOF={metrics['lof']:.2f}% esperado < 5%"
        assert metrics['r2'] > 0.95, f"R²={metrics['r2']:.3f} esperado > 0.95"

    def test_r2_per_sample_matches_individual_reconstruction(self):
        """r2_per_sample[j] deve refletir a qualidade do ajuste APENAS da
        amostra j (contra a própria média), não a média global do pool —
        varia sample a sample quando o SNR varia entre amostras."""
        n_samples = 8
        rng = np.random.default_rng(3)
        C_A = rng.uniform(1.0, 3.0, n_samples)
        # Ruído crescente por amostra — amostras finais devem ajustar pior.
        X = np.zeros((len(self.ppm), n_samples))
        for j in range(n_samples):
            noise_std = 0.01 + 0.15 * (j / (n_samples - 1))
            col = self.S_A * C_A[j] + rng.normal(0, noise_std, len(self.ppm))
            X[:, j] = np.clip(col, 0, None)

        S_init = self.S_A[:, np.newaxis]
        C, S, metrics = mcr_als(X, n_components=1, S_init=S_init,
                                max_iter=300, tol=1e-8)

        assert 'r2_per_sample' in metrics and 'r2_median' in metrics
        r2_ps = np.asarray(metrics['r2_per_sample'])
        assert r2_ps.shape == (n_samples,)
        assert np.all(np.isfinite(r2_ps))
        # Recomputar R² da amostra 0 (baixo ruído) manualmente e comparar —
        # confirma que cada entrada usa a própria média da amostra, não a
        # média global usada pelo 'r2' agregado.
        X_hat = S @ C.T
        x0, xh0 = X[:, 0], X_hat[:, 0]
        expected_r2_0 = 1.0 - np.sum((x0 - xh0) ** 2) / np.sum((x0 - x0.mean()) ** 2)
        assert r2_ps[0] == pytest.approx(expected_r2_0, abs=1e-9)
        # r2_median deve ser um valor intermediário do array por amostra.
        assert r2_ps.min() - 1e-9 <= metrics['r2_median'] <= r2_ps.max() + 1e-9
        # Amostra de baixo ruído deve ajustar melhor que a de alto ruído.
        assert r2_ps[0] > r2_ps[-1]


# ---------------------------------------------------------------------------
# Testes: robustez
# ---------------------------------------------------------------------------

class TestMCRALSRobustness:

    def setup_method(self):
        self.ppm = np.linspace(2.5, 1.5, 128)
        self.S_A = _lorentzian(self.ppm, 2.2, 0.005)
        self.S_B = _lorentzian(self.ppm, 1.8, 0.005)

    def test_single_sample_no_crash(self):
        """n_samples=1 não deve lançar excepção."""
        X = _build_mixture(self.ppm, [self.S_A, self.S_B], [[1.0], [1.0]])
        S_init = np.column_stack([self.S_A / (self.S_A.max() + 1e-12),
                                  self.S_B / (self.S_B.max() + 1e-12)])
        C, S, metrics = mcr_als(X, n_components=2, S_init=S_init)
        assert C.shape[0] == 1
        assert S.shape[1] == 2

    def test_no_negative_concentrations(self):
        """MCR (NNLS) deve garantir C ≥ 0."""
        n_samples = 10
        rng = np.random.default_rng(1)
        C_A = rng.uniform(0.1, 2.0, n_samples)
        C_B = rng.uniform(0.1, 2.0, n_samples)
        X = _build_mixture(self.ppm, [self.S_A, self.S_B], [C_A, C_B],
                           noise_std=0.01, rng=rng)
        S_init = np.column_stack([self.S_A, self.S_B])
        C, S, _ = mcr_als(X, n_components=2, S_init=S_init)
        assert np.all(C >= -1e-10), "Concentrações negativas encontradas"
        assert np.all(S >= -1e-10), "Perfis negativos encontrados"

    def test_one_component_trivial(self):
        """n_components=1: MCR deve recuperar o único perfil."""
        n_samples = 6
        C_A = np.ones(n_samples) * 2.0
        X = np.outer(self.S_A, C_A)
        S_init = self.S_A[:, np.newaxis]
        C, S, metrics = mcr_als(X, n_components=1, S_init=S_init, max_iter=200)
        assert metrics['r2'] > 0.90, f"R²={metrics['r2']:.3f} demasiado baixo para 1 componente"


# ---------------------------------------------------------------------------
# Testes: simulate_lorentzian_profile
# ---------------------------------------------------------------------------

class TestSimulateLorentzianProfile:

    def test_basic_singlet(self):
        ppm = np.linspace(3.0, 1.0, 256)
        sticks = [(2.0, 1.0)]   # formato: (ppm, rel_intensity)
        profile = simulate_lorentzian_profile(sticks, ppm, hwhm=0.005)
        assert profile.shape == ppm.shape
        assert profile.max() > 0
        # máximo deve estar próximo de 2.0 ppm
        assert abs(ppm[np.argmax(profile)] - 2.0) < 0.01

    def test_doublet_two_maxima(self):
        ppm = np.linspace(2.2, 1.8, 512)
        sticks = [(2.05, 1.0), (1.95, 1.0)]   # formato: (ppm, rel_intensity)
        profile = simulate_lorentzian_profile(sticks, ppm, hwhm=0.003)
        # Deve haver 2 máximos locais
        from scipy.signal import find_peaks
        peaks_idx, _ = find_peaks(profile, height=profile.max() * 0.3)
        assert len(peaks_idx) >= 2, f"Esperado ≥2 máximos, encontrado {len(peaks_idx)}"

    def test_non_negative(self):
        ppm = np.linspace(3.0, 1.0, 200)
        sticks = [(2.0, 1.0)]   # formato: (ppm, rel_intensity)
        profile = simulate_lorentzian_profile(sticks, ppm, hwhm=0.005)
        assert np.all(profile >= 0)
