//! Encodeur convolutif, champ de vitesse et déformation de la surface.

use burn::module::Module;
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig3d};
use burn::prelude::{Device, Tensor, TensorData};
use burn::tensor::activation::relu;
use surfoxide_burn::geometry::{flatten_channels, GridMap};
use surfoxide_core::Volume;
use surfoxide_ode::{integrate, Scheme};

use crate::data::N_CLASSES;

/// Canaux des trois échelles de l'encodeur (0,5, 1 et 2 mm pour des voxels de 0,5 mm).
pub const CHANNELS: [usize; 3] = [16, 32, 64];

fn conv(c_in: usize, c_out: usize, stride: usize, device: &Device) -> Conv3d {
    Conv3dConfig::new([c_in, c_out], [3, 3, 3])
        .with_stride([stride; 3])
        .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
        .init(device)
}

/// Encodeur à trois échelles : deux convolutions par échelle, passage à l'échelle suivante par un pas de 2.
#[derive(Module, Debug)]
pub struct Encoder {
    c1a: Conv3d,
    c1b: Conv3d,
    c2a: Conv3d,
    c2b: Conv3d,
    c3a: Conv3d,
    c3b: Conv3d,
}

impl Encoder {
    pub fn new(device: &Device) -> Self {
        let [a, b, c] = CHANNELS;
        Encoder {
            c1a: conv(N_CLASSES, a, 1, device),
            c1b: conv(a, a, 1, device),
            c2a: conv(a, b, 2, device),
            c2b: conv(b, b, 1, device),
            c3a: conv(b, c, 2, device),
            c3b: conv(c, c, 1, device),
        }
    }

    /// Caractéristiques des trois échelles pour une entrée `[1, N_CLASSES, nz, ny, nx]`.
    pub fn forward(&self, x: Tensor<5>) -> [Tensor<5>; 3] {
        let f1 = relu(self.c1b.forward(relu(self.c1a.forward(x))));
        let f2 = relu(self.c2b.forward(relu(self.c2a.forward(f1.clone()))));
        let f3 = relu(self.c3b.forward(relu(self.c3a.forward(f2.clone()))));
        [f1, f2, f3]
    }
}

/// Champ de vitesse : MLP sur les caractéristiques des trois échelles au point, et le temps.
#[derive(Module, Debug)]
pub struct Velocity {
    l1: Linear,
    l2: Linear,
    l3: Linear,
}

impl Velocity {
    pub fn new(device: &Device) -> Self {
        let n_in = CHANNELS.iter().sum::<usize>() + 1;
        Velocity {
            l1: LinearConfig::new(n_in, 128).init(device),
            l2: LinearConfig::new(128, 128).init(device),
            l3: LinearConfig::new(128, 3).init(device),
        }
    }

    pub fn forward(&self, features: Tensor<2>, t: f64) -> Tensor<2> {
        let n = features.dims()[0];
        let time = Tensor::<2>::ones([n, 1], &features.device()) * t;
        let h = relu(self.l1.forward(Tensor::cat(vec![features, time], 1)));
        let h = relu(self.l2.forward(h));
        self.l3.forward(h)
    }
}

/// Réseau de déformation de la surface blanche vers la piale.
#[derive(Module, Debug)]
pub struct DeformNet {
    pub encoder: Encoder,
    pub velocity: Velocity,
}

/// Caractéristiques d'un hémisphère, prêtes à être échantillonnées.
pub struct Features {
    flat: Vec<Tensor<2>>,
    maps: Vec<GridMap>,
}

/// Entrée de l'encodeur : classes en un-parmi-N, `[1, N_CLASSES, nz, ny, nx]`.
pub fn one_hot(classes: &Volume<u8>, device: &Device) -> Tensor<5> {
    let [nx, ny, nz] = classes.dims();
    let n = classes.len();
    let mut data = vec![0.0f32; N_CLASSES * n];
    for (i, &c) in classes.data().iter().enumerate() {
        data[c as usize * n + i] = 1.0;
    }
    Tensor::from_floats(TensorData::new(data, [1, N_CLASSES, nz, ny, nx]), device)
}

impl DeformNet {
    pub fn new(device: &Device) -> Self {
        DeformNet {
            encoder: Encoder::new(device),
            velocity: Velocity::new(device),
        }
    }

    /// Caractéristiques de l'hémisphère dont les classes sont `classes` (grille et affine du cadrage).
    pub fn features(&self, classes: &Volume<u8>, device: &Device) -> Features {
        let maps = [1usize, 2, 4].map(|s| GridMap::new(classes.dims(), classes.affine(), s, device));
        let levels = self.encoder.forward(one_hot(classes, device));
        Features {
            flat: levels.into_iter().map(flatten_channels).collect(),
            maps: maps.into(),
        }
    }

    /// Vitesse aux points `x` (`[M, 3]`, mm) à l'instant `t`.
    pub fn velocity_at(&self, f: &Features, x: Tensor<2>, t: f64) -> Tensor<2> {
        let sampled: Vec<Tensor<2>> = f
            .flat
            .iter()
            .zip(&f.maps)
            .map(|(flat, map)| map.sample(flat, &x))
            .collect();
        self.velocity.forward(Tensor::cat(sampled, 1), t)
    }

    /// Positions après intégration de 0 à 1 en `steps` pas RK4 depuis `x0`.
    pub fn deform(&self, f: &Features, x0: Tensor<2>, steps: usize) -> Tensor<2> {
        integrate(&|x, t| self.velocity_at(f, x, t), x0, 0.0, 1.0, steps, Scheme::Rk4)
    }
}
