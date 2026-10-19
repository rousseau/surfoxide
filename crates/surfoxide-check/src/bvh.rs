//! Hiérarchie de boîtes englobantes alignées sur les axes (AABB) sur les faces d'un maillage.

use surfoxide_core::vec3::Vec3;
use surfoxide_core::TriMesh;

/// Boîte englobante.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    pub fn of_points(points: &[Vec3]) -> Self {
        points.iter().fold(Self::EMPTY, |b, p| b.grow(*p))
    }

    pub fn grow(self, p: Vec3) -> Self {
        Aabb {
            min: std::array::from_fn(|a| self.min[a].min(p[a])),
            max: std::array::from_fn(|a| self.max[a].max(p[a])),
        }
    }

    pub fn union(self, o: Aabb) -> Self {
        Aabb {
            min: std::array::from_fn(|a| self.min[a].min(o.min[a])),
            max: std::array::from_fn(|a| self.max[a].max(o.max[a])),
        }
    }

    /// Intersection des boîtes fermées (le contact compte).
    pub fn overlaps(&self, o: &Aabb) -> bool {
        (0..3).all(|a| self.min[a] <= o.max[a] && o.min[a] <= self.max[a])
    }

    /// Distance au carré du point `p` à la boîte (nulle à l'intérieur).
    pub fn distance2(&self, p: Vec3) -> f64 {
        (0..3)
            .map(|a| {
                let d = (self.min[a] - p[a]).max(0.0).max(p[a] - self.max[a]);
                d * d
            })
            .sum()
    }

    fn centroid(&self) -> Vec3 {
        std::array::from_fn(|a| 0.5 * (self.min[a] + self.max[a]))
    }
}

#[derive(Debug, Clone)]
struct Node {
    bbox: Aabb,
    /// Feuille : faces `items[start..start + count]` ; nœud interne : enfants `start` et `start + 1`.
    start: u32,
    count: u32,
}

/// Hiérarchie de boîtes sur les faces d'un maillage (construite par coupe médiane selon l'axe le plus
/// étendu, au plus `LEAF` faces par feuille).
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<Node>,
    items: Vec<u32>,
    face_boxes: Vec<Aabb>,
}

const LEAF: usize = 4;

impl Bvh {
    pub fn new(mesh: &TriMesh) -> Self {
        let v = mesh.vertices();
        let face_boxes: Vec<Aabb> = mesh
            .faces()
            .iter()
            .map(|f| Aabb::of_points(&f.map(|i| v[i as usize])))
            .collect();
        let mut items: Vec<u32> = (0..mesh.n_faces() as u32).collect();
        let mut nodes = vec![Node {
            bbox: Aabb::EMPTY,
            start: 0,
            count: 0,
        }];
        if !items.is_empty() {
            Self::build(&mut nodes, 0, &mut items, 0, &face_boxes);
        }
        Bvh {
            nodes,
            items,
            face_boxes,
        }
    }

    fn build(nodes: &mut Vec<Node>, node: usize, items: &mut [u32], offset: usize, boxes: &[Aabb]) {
        let bbox = items.iter().fold(Aabb::EMPTY, |b, &i| b.union(boxes[i as usize]));
        if items.len() <= LEAF {
            nodes[node] = Node {
                bbox,
                start: offset as u32,
                count: items.len() as u32,
            };
            return;
        }
        let centroids = items
            .iter()
            .fold(Aabb::EMPTY, |b, &i| b.grow(boxes[i as usize].centroid()));
        let axis = (0..3)
            .max_by(|&a, &b| (centroids.max[a] - centroids.min[a]).total_cmp(&(centroids.max[b] - centroids.min[b])))
            .unwrap();
        let mid = items.len() / 2;
        items.select_nth_unstable_by(mid, |&a, &b| {
            boxes[a as usize].centroid()[axis]
                .total_cmp(&boxes[b as usize].centroid()[axis])
                .then(a.cmp(&b))
        });
        let left = nodes.len();
        nodes.push(Node {
            bbox: Aabb::EMPTY,
            start: 0,
            count: 0,
        });
        nodes.push(Node {
            bbox: Aabb::EMPTY,
            start: 0,
            count: 0,
        });
        nodes[node] = Node {
            bbox,
            start: left as u32,
            count: 0,
        };
        let (l, r) = items.split_at_mut(mid);
        Self::build(nodes, left, l, offset, boxes);
        Self::build(nodes, left + 1, r, offset + mid, boxes);
    }

    pub fn face_box(&self, f: u32) -> &Aabb {
        &self.face_boxes[f as usize]
    }

    /// Appelle `visit` pour chaque face dont la boîte rencontre `query`.
    pub fn query_box(&self, query: &Aabb, mut visit: impl FnMut(u32)) {
        if self.items.is_empty() {
            return;
        }
        let mut stack = vec![0usize];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !node.bbox.overlaps(query) {
                continue;
            }
            if node.count > 0 {
                for &f in &self.items[node.start as usize..(node.start + node.count) as usize] {
                    if self.face_boxes[f as usize].overlaps(query) {
                        visit(f);
                    }
                }
            } else {
                stack.push(node.start as usize);
                stack.push(node.start as usize + 1);
            }
        }
    }

    /// Plus proche face de `p` selon `dist2(face) -> distance au carré` ; rend (face, distance au carré).
    pub fn nearest(&self, p: Vec3, mut dist2: impl FnMut(u32) -> f64) -> Option<(u32, f64)> {
        if self.items.is_empty() {
            return None;
        }
        let mut best = (u32::MAX, f64::INFINITY);
        let mut stack = vec![(0usize, self.nodes[0].bbox.distance2(p))];
        while let Some((n, d)) = stack.pop() {
            if d >= best.1 {
                continue;
            }
            let node = &self.nodes[n];
            if node.count > 0 {
                for &f in &self.items[node.start as usize..(node.start + node.count) as usize] {
                    if self.face_boxes[f as usize].distance2(p) < best.1 {
                        let d = dist2(f);
                        if d < best.1 {
                            best = (f, d);
                        }
                    }
                }
            } else {
                let (a, b) = (node.start as usize, node.start as usize + 1);
                let (da, db) = (self.nodes[a].bbox.distance2(p), self.nodes[b].bbox.distance2(p));
                // Le plus proche est dépilé en premier.
                if da < db {
                    stack.push((b, db));
                    stack.push((a, da));
                } else {
                    stack.push((a, da));
                    stack.push((b, db));
                }
            }
        }
        Some(best)
    }
}
