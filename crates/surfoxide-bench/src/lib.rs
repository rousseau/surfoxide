//! `surfoxide-bench` : fantômes à vérité connue pour juger la chaîne de surfoxide (plan qualité, phase 0).
//!
//! Fantômes A : un couple blanche/piale vrai (atlas fœtal dHCP, ou toute autre surface fermée) est voxelisé en
//! étiquettes BOUNTI-19, fractions de volume partiel et T2w synthétique ; les champs vrais (épaisseur, courbure,
//! D*) sont calculés sur la blanche vraie. Un bruit de contour optionnel déplace les surfaces le long de leurs
//! normales avant la voxelisation ; la vérité reste la surface non déplacée.
//!
//! Sans Burn ni format de fichier : la lecture et l'écriture sont faites par `surfx phantom`.

pub mod analytic;
pub mod compare;
pub mod image;
pub mod noise;
pub mod rng;
pub mod truth;
pub mod voxel;

use surfoxide_core::{Hemisphere, TriMesh};

pub use image::T2wParams;
pub use rng::Rng;
pub use truth::Truth;
pub use voxel::Grid;

/// Paramètres d'un fantôme A.
#[derive(Debug, Clone, PartialEq)]
pub struct PhantomParams {
    /// Côté d'un voxel (mm).
    pub voxel: f64,
    /// Marge autour de la piale (mm).
    pub margin: f64,
    /// Moyenne quadratique du bruit de contour (mm) ; 0 : surfaces voxelisées telles quelles.
    pub noise_sigma: f64,
    /// Longueur de corrélation du bruit de contour (mm).
    pub noise_length: f64,
    /// Graine de tous les tirages.
    pub seed: u64,
    /// Déplacement minimal de la piale le long de la normale de la blanche pour qu'un sommet soit mesuré (mm).
    pub min_offset: f64,
    pub t2w: T2wParams,
}

impl Default for PhantomParams {
    fn default() -> Self {
        Self {
            voxel: 0.5,
            margin: 3.0,
            noise_sigma: 0.0,
            noise_length: 1.5,
            seed: 0,
            min_offset: 0.1,
            t2w: T2wParams::default(),
        }
    }
}

/// Flux du générateur, un par usage.
mod streams {
    pub const WHITE: u64 = 1;
    pub const PIAL: u64 = 2;
    pub const BIAS: u64 = 3;
    pub const IMAGE: u64 = 4;
}

/// Fantôme A : volumes sur `grid` (x le plus rapide) et vérité.
#[derive(Debug, Clone)]
pub struct Phantom {
    pub grid: Grid,
    /// Segmentation BOUNTI-19.
    pub labels: Vec<i16>,
    /// Fractions de volume partiel : blanche, cortex, LCR.
    pub fractions: [Vec<f32>; 3],
    pub t2w: Vec<f32>,
    /// Surfaces voxelisées (déplacées par le bruit de contour).
    pub white_noisy: TriMesh,
    pub pial_noisy: TriMesh,
    pub truth: Truth,
}

/// Fantôme A du couple vrai (`white`, `pial`) de l'hémisphère `hemi` ; `medial` marque les sommets de la face
/// médiale de la blanche (exclus des champs vrais).
pub fn phantom(white: &TriMesh, pial: &TriMesh, medial: &[bool], hemi: Hemisphere, p: &PhantomParams) -> Phantom {
    let noisy = |mesh: &TriMesh, stream| {
        let mut rng = Rng::stream(p.seed, stream);
        let f = noise::smooth_field(mesh, p.noise_sigma, p.noise_length, &mut rng);
        noise::displace(mesh, &f)
    };
    let white_noisy = noisy(white, streams::WHITE);
    let pial_noisy = noisy(pial, streams::PIAL);
    let grid = Grid::around(&[&white_noisy, &pial_noisy], p.voxel, p.margin);
    let counts = voxel::counts(&grid, &white_noisy, &pial_noisy);
    let t2w = image::t2w(
        &grid,
        &counts,
        &p.t2w,
        &mut Rng::stream(p.seed, streams::BIAS),
        &mut Rng::stream(p.seed, streams::IMAGE),
    );
    Phantom {
        labels: image::labels(&counts, hemi),
        fractions: image::fractions(&counts),
        t2w,
        truth: truth::truth(white, pial, medial, p.min_offset),
        grid,
        white_noisy,
        pial_noisy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    fn scaled(m: &TriMesh, s: [f64; 3]) -> TriMesh {
        TriMesh::new(
            m.vertices()
                .iter()
                .map(|p| [p[0] * s[0], p[1] * s[1], p[2] * s[2]])
                .collect(),
            m.faces().to_vec(),
        )
        .unwrap()
    }

    /// Volume de la blanche par ses fractions, contre le volume analytique de la boule ou de l'ellipsoïde.
    fn volume_check(axes: [f64; 3], thickness: f64) {
        let s = icosphere(5, 1.0);
        let white = scaled(&s, axes);
        let pial = scaled(&s, axes.map(|a| a + thickness));
        let medial = vec![false; white.n_vertices()];
        let p = PhantomParams {
            voxel: 0.5,
            ..Default::default()
        };
        let ph = phantom(&white, &pial, &medial, Hemisphere::Left, &p);
        let v3 = p.voxel.powi(3);
        let vw: f64 = ph.fractions[0].iter().map(|&f| f as f64).sum::<f64>() * v3;
        let vp: f64 = ph.fractions[0]
            .iter()
            .zip(&ph.fractions[1])
            .map(|(&a, &b)| (a + b) as f64)
            .sum::<f64>()
            * v3;
        let ball = |a: [f64; 3]| 4.0 / 3.0 * std::f64::consts::PI * a[0] * a[1] * a[2];
        let (rw, rp) = (vw / ball(axes) - 1.0, vp / ball(axes.map(|a| a + thickness)) - 1.0);
        assert!(
            rw.abs() < 0.01 && rp.abs() < 0.01,
            "écarts relatifs {rw} et {rp} ({axes:?})"
        );
    }

    #[test]
    fn boule_volume_des_fractions_a_un_pour_cent() {
        volume_check([10.0; 3], 1.5);
    }

    #[test]
    fn ellipsoide_volume_des_fractions_a_un_pour_cent() {
        volume_check([12.0, 18.0, 9.0], 1.5);
    }

    #[test]
    fn meme_graine_meme_fantome_autre_graine_autre_fantome() {
        let s = icosphere(3, 1.0);
        let white = scaled(&s, [8.0, 10.0, 7.0]);
        let pial = scaled(&s, [9.5, 11.5, 8.5]);
        let medial = vec![false; white.n_vertices()];
        let p = PhantomParams {
            voxel: 0.6,
            noise_sigma: 0.2,
            seed: 11,
            ..Default::default()
        };
        let a = phantom(&white, &pial, &medial, Hemisphere::Right, &p);
        let b = phantom(&white, &pial, &medial, Hemisphere::Right, &p);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.t2w, b.t2w);
        assert_eq!(a.fractions, b.fractions);
        assert_eq!(a.white_noisy.vertices(), b.white_noisy.vertices());
        let c = phantom(
            &white,
            &pial,
            &medial,
            Hemisphere::Right,
            &PhantomParams { seed: 12, ..p },
        );
        assert_ne!(a.t2w, c.t2w);
        assert_ne!(a.white_noisy.vertices(), c.white_noisy.vertices());
    }
}
