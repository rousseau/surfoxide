//! Générateur pseudo-aléatoire déterministe : xoshiro256** (Blackman et Vigna), état initialisé par SplitMix64.
//!
//! Écrit ici plutôt que tiré d'un crate externe : quelques lignes, et une suite de tirages figée pour une graine
//! donnée, quelle que soit la machine (critère (i) du banc : même graine → fichiers identiques).

/// Générateur xoshiro256**, avec une normale gaussienne de réserve (Box-Muller tire deux valeurs à la fois).
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Générateur de la graine `seed`.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Self {
            s: [0; 4].map(|_| splitmix64(&mut x)),
            spare: None,
        }
    }

    /// Flux indépendant numéro `stream` de la graine `seed` (un flux par usage : bruit de chaque surface, biais, bruit
    /// d'image), pour qu'ajouter un tirage à un usage ne change pas les autres.
    pub fn stream(seed: u64, stream: u64) -> Self {
        let mut x = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Self::new(splitmix64(&mut x))
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniforme sur [0, 1), 53 bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Normale centrée réduite (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // 1 − u ∈ (0, 1] : le logarithme reste fini.
        let r = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        let theta = std::f64::consts::TAU * self.uniform();
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meme_graine_meme_suite_et_flux_distincts() {
        let a: Vec<u64> = (0..8).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        let b: Vec<u64> = (0..8).scan(Rng::new(7), |r, _| Some(r.next_u64())).collect();
        assert_eq!(a, b);
        let c: Vec<u64> = (0..8).scan(Rng::stream(7, 1), |r, _| Some(r.next_u64())).collect();
        assert_ne!(a, c);
    }

    #[test]
    fn normale_centree_reduite() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let x: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let m = x.iter().sum::<f64>() / n as f64;
        let v = x.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
        assert!(m.abs() < 0.01, "moyenne {m}");
        assert!((v - 1.0).abs() < 0.01, "variance {v}");
    }
}
