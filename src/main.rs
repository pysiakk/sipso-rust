use rand::{Rng, prelude::ThreadRng, seq::SliceRandom};
use optlib_testfunc::*;
#[derive(Debug)]
struct Particle{
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_score: f64,
    neighbors: Vec<usize>,
}

fn initialize_particles(n_particles: usize, bounds: Vec<(f64, f64)>, target_function: fn(&Vec<f64>) -> f64) -> Vec<Particle>{
    let mut particles: Vec<Particle> = Vec::with_capacity(n_particles);
    let n_dim: usize = bounds.len();
    let mut rng: ThreadRng = rand::thread_rng();
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
                best_score: target_function(&position),
                neighbors: Vec::new(),
            }
        );
    };
    particles
}

fn _neigh_full(mut particles: Vec<Particle>) -> Vec<Particle>{
    let n_particles: usize = particles.len();
    for i in 0..n_particles{
        for j in 0..n_particles{
            if i != j{
                particles[i].neighbors.push(j);
            }
        }
    }
    particles
}

fn _neigh_ba(mut particles: Vec<Particle>, m: usize, m0: usize) -> Vec<Particle>{
    let n_particles: usize = particles.len();
    let mut repeated: Vec<usize> = Vec::new();
    let mut m0 = m0;
    if n_particles < m0{
        m0 = n_particles;
    }
    for i in 0..m0{
        for j in 0..m0{
            if j != i{
                particles[i].neighbors.push(j);
                repeated.push(j)
            }
        }
    }
    let mut rng = rand::thread_rng();
    for i in m0..n_particles{
        let choice: Vec<usize> = repeated.choose_multiple(&mut rng, m).cloned().collect();
        for sample in choice{
            particles[i].neighbors.push(sample);
            particles[sample].neighbors.push(i);
            repeated.push(i);
            repeated.push(sample);
        }
    }
    particles
}

fn _update_best(mut particles: Vec<Particle>, id: usize, target_function: fn(&Vec<f64>) -> f64) -> Vec<Particle>{
    let new_score = target_function(&particles[id].position);
    if new_score < particles[id].best_score{
        particles[id].best_score = new_score;
        particles[id].best_position = particles[id].position.clone();
    }
    particles
}

fn _full_update(mut particles: Vec<Particle>, id: usize, constriction_coef: f64, acceleration_coef: f64, target_function: fn(&Vec<f64>) -> f64) -> Vec<Particle>{
    let n_dim = particles[id].position.len();
    let degree = particles[id].neighbors.len();
    let mut new_velocity: Vec<f64> = vec![0.0; n_dim];
    let mut new_position: Vec<f64> = Vec::with_capacity(n_dim);
    let mut rng = rand::thread_rng();
    for i in 0..degree{
        let neigh = particles[id].neighbors[i];
        let c = rng.gen::<f64>();
        for d in 0..n_dim{
            new_velocity[d] += acceleration_coef * c * (particles[neigh].best_position[d] - particles[id].position[d]);
        }
    }
    for d in 0..n_dim{
        new_velocity[d] = constriction_coef * (particles[id].velocity[d] + (new_velocity[d] / degree as f64));
        new_position.push(particles[id].position[d] + new_velocity[d]);
    }
    particles[id].velocity = new_velocity;
    particles[id].position = new_position;
    _update_best(particles, id, target_function)
}

fn _get_best_neighbor(particles: &Vec<Particle>, id: usize) -> Vec<f64>{
    let n_neighbors: usize = particles[id].neighbors.len();
    let first_neighbor: usize = particles[id].neighbors[0];
    let mut gbest_score: f64 = particles[first_neighbor].best_score.clone();
    let mut gbest: Vec<f64> = particles[first_neighbor].best_position.clone();
    for i in 1..n_neighbors{
        let neighbor: usize = particles[id].neighbors[i];
        if gbest_score > particles[neighbor].best_score{
            gbest_score = particles[neighbor].best_score;
            gbest = particles[neighbor].best_position.clone();
        }
    }
    gbest
}

fn _single_update(mut particles: Vec<Particle>, id: usize, constriction_coef: f64, nostalgia_coef: f64, social_coef: f64, target_function: fn(&Vec<f64>) -> f64) -> Vec<Particle>{
    let n_dim = particles[id].position.len();
    let mut rng: ThreadRng = rand::thread_rng();
    let c1: f64 = rng.gen::<f64>() * nostalgia_coef;
    let c2: f64 = rng.gen::<f64>() * social_coef;
    let gbest: Vec<f64> = _get_best_neighbor(&particles, id);
    for d in 0..n_dim{
        let nostalgia: f64 = c1 * (particles[id].best_position[d] - particles[id].position[d]);
        let social: f64 = c2 * (gbest[d] - particles[id].position[d]);
        particles[id].velocity[d] = constriction_coef * (particles[id].velocity[d] + nostalgia + social);
        particles[id].position[d] += particles[id].velocity[d];
    }
    _update_best(particles, id, target_function)
}

fn _get_gbest(particles: &Vec<Particle>) -> (f64, Vec<f64>){
    let mut gbest = particles[0].best_position.clone();
    let mut gbest_score = particles[0].best_score;
    for i in 1..(particles.len()){
        if gbest_score > particles[i].best_score{
            gbest_score = particles[i].best_score;
            gbest = particles[i].best_position.clone();
        }
    }
    (gbest_score, gbest)
}

fn run_pso(
    n_iter: usize, n_particles: usize, bounds: Vec<(f64, f64)>, target_function: fn(&Vec<f64>) -> f64, k: usize,
    constriction_coef: f64, nostalgia_coef: f64, social_coef: f64, neighborhood: &str, verbose: usize) -> (f64, Vec<f64>){
    
        let mut particles = initialize_particles(n_particles, bounds, target_function);
        if neighborhood == "ba" {
            particles = _neigh_ba(particles, 2, 4);
        } else if neighborhood == "full" {
            particles = _neigh_full(particles);            
        }
        for iter in 0..n_iter{
            for i in 0..n_particles{
                if particles[i].neighbors.len() > k{
                    particles = _full_update(particles, i, constriction_coef, nostalgia_coef + social_coef, target_function)
                } else {
                    particles = _single_update(particles, i, constriction_coef, nostalgia_coef, social_coef, target_function);
                }
            }
            if verbose == 2 && iter%100 == 0{
                println!("Iter {}: {:#?}", iter, _get_gbest(&particles).0);
            } 
        }
        if verbose == 1 || verbose == 2{
            println!("Neigh: {} | k: {} | score: {}", neighborhood, k, _get_gbest(&particles).0);
        }
        _get_gbest(&particles)
}
fn main() {
    let n_iter: usize = 5000;
    let n_particles: usize = 50;    
    let n_dim: usize = 30;
    let rastrigin_bounds: Vec<(f64, f64)> = vec![(-5.12, 5.12); n_dim];
    let rosenbrock_bounds: Vec<(f64, f64)> = vec![(-30.0, 30.0); n_dim];
    let k: usize = 5;
    let target_function: fn(&Vec<f64>) -> f64 = optlib_testfunc::rastrigin;
    let constriction_coef: f64 = 0.7298;
    let nostalgia_coef: f64 = 2.05;
    let social_coef: f64 = 2.05;
    let verbose: usize = 1;
    
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 0, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 5, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 50, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 0, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 5, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 50, constriction_coef, nostalgia_coef, social_coef, "ba", verbose);
    
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 0, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 5, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    run_pso(n_iter, n_particles, rastrigin_bounds.clone(), rastrigin, 50, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 0, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 5, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    run_pso(n_iter, n_particles, rosenbrock_bounds.clone(), rosenbrock, 50, constriction_coef, nostalgia_coef, social_coef, "full", verbose);
    // let mut particles = initialize_particles(n_particles, bounds, target_function);
    // particles = _neigh_ba(particles, 2, 4);
    // let mut degree: Vec<usize> = Vec::new();
    // for particle in particles.iter() {
    //     degree.push(particle.neighbors.len());
    // }
    // println!("{:#?}", degree);
}
