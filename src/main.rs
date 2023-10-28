/*
In questo momento il codice è statico, bisogna renderlo dinamico
usando la memoria dinamica in modo da poter automaticamente allocare
un quantitativo corretto di memoria e non un valore predefinito.
*/
use fast_math::exp;
use rand::Rng;
use raylib::prelude::*;

const NUM_TEST: u32 = 50 * 1000;
const EPS: f64 = 0.1;
const RATE: f64 = 1.7;

const INPUT: usize = 2;
const NUM_TRAIN_SAMPLE: usize = 4;
const AND: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
];

const OR: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

const XOR: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
];

const NAND: [[f64; 3]; NUM_TRAIN_SAMPLE] = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
];

const DATA: [[f64; 3]; NUM_TRAIN_SAMPLE] = XOR;
const LEN_ARC: usize = 2;
const ARC: [usize; LEN_ARC] = [2, 1];
const NUM_NEURONS: usize = 3;

const SCREEN_HEIGHT: i32 = 720;
const SCREEN_WIDTH: i32 = (SCREEN_HEIGHT * 16) / 9;

const GRAPHICS_WIDTH: i32 = 600;
const GRAPHICS_HEIGHT: i32 = 400;

const TEXT_DIM: i32 = 23;
const DEF_PADDING: i32 = 30;

#[derive(Debug, Default)]
struct Neuron {
    w: Vec<f64>,
    b: f64,
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(SCREEN_WIDTH, SCREEN_HEIGHT)
        .title("Hello, World")
        .build();

    let mut nrs: [Neuron; NUM_NEURONS] = Default::default();
    let mut cont = 0;
    let mut cost: f64;
    let mut costs_vec: Vec<f64> = Vec::new();

    for layout_idx in 0..LEN_ARC {
        for _ in 0..ARC[layout_idx] {
            nrs[cont] = get_randomize_neuron(layout_idx);
            cont += 1;
        }
    }

    //println!("{:#?}", nrs);

    let mut cont: u32 = 0;
    let mut max_cost: f64 = -1.0;

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::BLACK);
        //d.draw_text("Hello, world!", 12, 12, 20, Color::WHITE);

        if cont < NUM_TEST {
            cost = update_weights(&mut nrs, 0);

            //println!("{}", cost);
            if cont == 0 {
                max_cost = cost;
                costs_vec.push(cost);
            } else {
                costs_vec.push((max_cost - cost) / max_cost);
                //println!("{} {} {}", cost, max_cost, (max_cost - cost) / max_cost);
            }
        }

        let vec_len = costs_vec.len();

        for n_point in 1..vec_len {
            d.draw_circle(
                (((n_point as f64) / (vec_len as f64)) * GRAPHICS_WIDTH as f64) as i32 + 20,
                (costs_vec[n_point] * GRAPHICS_HEIGHT as f64) as i32 + 20,
                2.0,
                Color::RED,
            );
        }

        cont += 1;

        for x in 0..NUM_TRAIN_SAMPLE {
            d.draw_text(
                &(DATA[x][0].to_string()),
                (0.75 * SCREEN_WIDTH as f64) as i32,
                ((0.25 * SCREEN_HEIGHT as f64) as i32) + (x as i32 * DEF_PADDING),
                TEXT_DIM,
                Color::WHITE,
            );

            d.draw_text(
                &(DATA[x][1].to_string()),
                (0.75 * SCREEN_WIDTH as f64) as i32 + DEF_PADDING,
                ((0.25 * SCREEN_HEIGHT as f64) as i32) + (x as i32 * DEF_PADDING),
                TEXT_DIM,
                Color::WHITE,
            );

            d.draw_text(
                &(DATA[x][2].to_string()),
                (0.75 * SCREEN_WIDTH as f64) as i32 + (DEF_PADDING * 2),
                ((0.25 * SCREEN_HEIGHT as f64) as i32) + (x as i32 * DEF_PADDING),
                TEXT_DIM,
                Color::WHITE,
            );

            let res = format!(
                "{:.4}",
                nn_get_result(&mut nrs, [DATA[x][0], DATA[x][1]], false)
            );

            d.draw_text(
                &res,
                (0.75 * SCREEN_WIDTH as f64) as i32 + (DEF_PADDING * 3),
                ((0.25 * SCREEN_HEIGHT as f64) as i32) + (x as i32 * DEF_PADDING),
                TEXT_DIM,
                Color::WHITE,
            );
        }
    }
}

fn sigmuid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x as f32)) as f64
}

fn randomf() -> f64 {
    ((rand::thread_rng().gen_range(0..=1000) as f64) / 1000.0)
}

fn cost(nrs: &mut [Neuron; NUM_NEURONS]) -> f64 {
    let mut out: f64 = 0.0;
    let mut res: f64 = -1.0;

    for sample_idx in 0..NUM_TRAIN_SAMPLE {
        //Per ogni tupla d'input
        let mut neuron_idx: usize = 0;
        let mut output: Vec<f64> = Vec::with_capacity(ARC[0]);

        for layer_idx in 0..LEN_ARC {
            //Per ogni Layer dell'architettura

            let mut input: Vec<f64> = Vec::with_capacity(output.len());
            input = output.clone();
            if layer_idx != 0 {
                output = Vec::with_capacity(ARC[layer_idx]);
            }

            for _ in 0..(ARC[layer_idx]) {
                //Per ogni Neurone del Layer
                let mut somma = 0.0;
                for inp_idx in 0..((nrs[neuron_idx].w).len()) {
                    //Per ogni input
                    //println!("{} {} {} {}", idx, d, nrs.len(), nrs[0].w.len());
                    if layer_idx == 0 {
                        somma += nrs[neuron_idx].w[inp_idx] * DATA[sample_idx][inp_idx];
                    } else {
                        somma += nrs[neuron_idx].w[inp_idx] * input[inp_idx];
                    }
                }
                somma += nrs[neuron_idx].b;
                output.push(sigmuid(somma));
                neuron_idx += 1;
            }
            res = output[0];
        }
        let dst: f64 = res - DATA[sample_idx][INPUT];
        out += dst * dst;
    }

    out / (NUM_TRAIN_SAMPLE as f64)
}

fn get_randomize_neuron(lidx: usize) -> Neuron {
    let mut n: Neuron = Neuron {
        w: Vec::new(),
        b: 0.0,
    };
    if lidx == 0 {
        n.w = Vec::with_capacity(INPUT);
        for _ in 0..INPUT {
            (n.w).push(randomf());
        }
    } else {
        n.w = Vec::with_capacity(ARC[(lidx - 1) as usize]);
        for _ in 0..ARC[(lidx - 1) as usize] {
            (n.w).push(randomf());
        }
    }

    n.b = randomf();
    n
}

fn update_weights(nrs: &mut [Neuron; NUM_NEURONS], pr: u8) -> f64 {
    let cst: f64 = cost(nrs);

    let mut idx: usize = 0;

    for layer_idx in 0..LEN_ARC {
        //Per ogni Layer dell'architettura
        for _ in 0..ARC[layer_idx] {
            //Per ogni Neurone del Layer
            let weight: Vec<f64> = (nrs[idx].w).clone();
            let bias: f64 = nrs[idx].b;
            for inp_idx in 0..((nrs[idx].w).len()) {
                //Per ogni input

                nrs[idx].w[inp_idx] += EPS;
                let dw: f64 = (cost(nrs) - cst) / EPS;
                nrs[idx].w[inp_idx] = weight[inp_idx];
                nrs[idx].w[inp_idx] -= dw * RATE;
            }

            nrs[idx].b += EPS;
            let dw: f64 = (cost(nrs) - cst) / EPS;
            nrs[idx].b = bias;
            nrs[idx].b -= dw * RATE;
            idx += 1;
        }
    }

    if pr == 1 {
        println!("-----------------------");
        for i in 0..=1 {
            for k in 0..=1 {
                nn_get_result(nrs, [i as f64, k as f64], true);
            }
        }
        println!("------------------------");
    }

    cst
}

//stampa a terminale il risultato e lo ritorna anche
fn nn_get_result(nrs: &mut [Neuron; NUM_NEURONS], data: [f64; INPUT], print: bool) -> f64 {
    let mut idx: usize = 0;
    let mut res: f64 = -1.0;
    let mut output: Vec<f64> = Vec::with_capacity(ARC[0]);

    for b in 0..LEN_ARC {
        //Per ogni Layer dell'architettura
        /*
        In questa parte andiamo a copiare il risultato del output che
        diventerà l'input per la prossima serie di neuroni. In più se
        non siamo alla prima simulazione il vettore output verrà
        reinizializzato
        */
        let mut input: Vec<f64> = Vec::with_capacity(output.len());
        input = output.clone();
        if b != 0 {
            output = Vec::with_capacity(ARC[b]);
        }

        for _ in 0..(ARC[b]) {
            //Per ogni Neurone del Layer
            let mut somma = 0.0;
            for d in 0..((nrs[idx].w).len()) {
                //Per ogni input
                //println!("{} {} {} {} {}", idx, a, b, c, d);
                if b == 0 {
                    somma += nrs[idx].w[d] * data[d];
                } else {
                    somma += nrs[idx].w[d] * input[d];
                }
            }
            somma += nrs[idx].b;
            output.push(sigmuid(somma));
            idx += 1;
        }
        res = output[0];
    }

    if print {
        println!("{} {} {}", data[0], data[1], res);
    }

    res
}
