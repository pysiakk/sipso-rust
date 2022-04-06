use rand::Rng;

#[derive(Debug)]
struct Particle{
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_score: f64,
}

fn initialize_particles(n_particles: usize, n_dim: usize, bounds: Vec<(f64, f64)>) -> Vec<Particle>{
    let mut particles: Vec<Particle> = Vec::with_capacity(n_particles);
    let mut rng = rand::thread_rng();
    for _ in 0..n_particles{
        let mut position: Vec<f64> = Vec::with_capacity(n_dim);
        for j in 0..n_dim{
            position.push(rng.gen_range((bounds[j].0)..(bounds[j].1)));
        }
        let mut velocity: Vec<f64> = Vec::with_capacity(n_dim);
        for j in 0..n_dim{
            velocity.push(rng.gen_range((bounds[j].0)..(bounds[j].1)));
        }
        particles.push(
            Particle{
                position: position.clone(),
                velocity: velocity,
                best_position: position.clone(),
                best_score: 1f64,
            }
        );
    };
    particles
}

fn main() {
    let mut particles = initialize_particles(1, 2, vec![(-1.0, 1.0), (-2.0, 2.0), (-3.1, 3.6)]);
    particles[0].best_position[0] = 1000000000000000000f64;
    println!("{:#?}", particles);
}
