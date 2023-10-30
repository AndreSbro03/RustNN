use fast_math::exp;
use rand::Rng;
use raylib::prelude::*;

//INIZIO COSTANTI NN
const NUM_TEST: u32 = 25 * 1000;
const EPS: f64 = 0.3;
const MAX_RATE: f64 = 5.0;

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
const LEN_ARC: usize = 4;
const ARC: [usize; LEN_ARC] = [5, 5, 5, 5];
const NUM_NEURONS: usize = 20;

//FINE COSTANTI NN

//INIZIO COSTANTI PER RAYLIB
const SCREEN_HEIGHT: i32 = 720;
const SCREEN_WIDTH: i32 = (SCREEN_HEIGHT * 16) / 9;

//COSTANTI BASE
const TEXT_DIM: i32 = 23;
const DEF_PADDING: i32 = 30;
const DEF_MARGIN: i32 = 8;

//GESTIONE DELLA BARRA SUPERIORE
const TOP_BAR_POS: IntVec2 = IntVec2{x: 75, y: 40};
const TOP_BAR_DIM: IntVec2 = IntVec2{x: SCREEN_WIDTH / 4 * 3, y: 0};

//GESTIONE DELLA BARRA DEL RATE
const RATE_BAR_H_PADDING: i32 = SCREEN_WIDTH / 16;
const RATE_BAR_POS: IntVec2 = IntVec2{x: RATE_BAR_H_PADDING, y: SCREEN_HEIGHT / 9 + TOP_BAR_POS.y + TOP_BAR_DIM.y};
const RATE_BAR_DIM: IntVec2 = IntVec2{x: SCREEN_WIDTH - 2 * RATE_BAR_H_PADDING, y: 0}; 
const RATE_BAR_RADIUS: f32 = 15.0;

//GESTIONE DEL GRAFICO
const GRAPHIC_POS: IntVec2 = IntVec2{x: RATE_BAR_POS.x, y: RATE_BAR_POS.y + DEF_PADDING * 2};
const GRAPHIC_DIM: IntVec2 = IntVec2{x: 600, y: 400};


#[derive(Debug, Default)]
struct Neuron {
    w: Vec<f64>,
    b: f64,
}

#[derive(Debug, Default)]
struct IntVec2 {
    x: i32,
    y: i32,
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

    let mut rate: f64 = 1.7;

    let mut cont: u32 = 0;
    let mut max_cost: f64 = -1.0;

    let mut rate_circle: IntVec2 = IntVec2 { x: ((rate / MAX_RATE) * (SCREEN_WIDTH - 2 * RATE_BAR_H_PADDING) as f64)
        as i32
        + RATE_BAR_H_PADDING, y: RATE_BAR_POS.y };

    let mut is_rate_button_clicked = false;

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::BLACK);

        if cont < NUM_TEST {
            cost = update_weights(&mut nrs, rate, false);

            //println!("{}", cost);
            if cont == 0 {
                max_cost = cost;
                costs_vec.push(cost);
            } else {
                costs_vec.push((max_cost - cost) / max_cost);
                //println!("{} {} {}", cost, max_cost, (max_cost - cost) / max_cost);
            }
            cont += 1;
        }

        nn_draw_graph(&mut d, &mut costs_vec);
        nn_draw_lg_text(&mut d, &mut nrs);
        nn_draw_rate_bar(&mut d, &mut rate_circle, &mut is_rate_button_clicked, &mut rate);
        nn_draw_infos(&mut d, rate, cont);
        //println!("{:#?}", nrs);
    }
    
}

fn sigmuid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x as f32)) as f64
}

fn randomf() -> f64 {
    (rand::thread_rng().gen_range(-1000..=1000) as f64) / 1000.0 * 5.0
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

fn update_weights(nrs: &mut [Neuron; NUM_NEURONS], rate: f64, print: bool) -> f64 {
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
                nrs[idx].w[inp_idx] -= dw * rate;
            }

            nrs[idx].b += EPS;
            let dw: f64 = (cost(nrs) - cst) / EPS;
            nrs[idx].b = bias;
            nrs[idx].b -= dw * rate;
            idx += 1;
        }
    }

    if print == true {
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

fn nn_draw_lg_text(d: &mut RaylibDrawHandle<'_>, nrs: &mut [Neuron; NUM_NEURONS]) {
    let abs_x = (0.75 * SCREEN_WIDTH as f64) as i32;
    let abs_y = (0.25 * SCREEN_HEIGHT as f64) as i32;

    for x in 0..NUM_TRAIN_SAMPLE {
        d.draw_text(
            &(DATA[x][0].to_string()),
            abs_x,
            abs_y + (x as i32 * DEF_PADDING),
            TEXT_DIM,
            Color::WHITE,
        );

        d.draw_text(
            &(DATA[x][1].to_string()),
            abs_x + DEF_PADDING,
            abs_y + (x as i32 * DEF_PADDING),
            TEXT_DIM,
            Color::WHITE,
        );

        d.draw_text(
            &(DATA[x][2].to_string()),
            abs_x + (DEF_PADDING * 2),
            abs_y + (x as i32 * DEF_PADDING),
            TEXT_DIM,
            Color::WHITE,
        );

        let res = format!("{:.4}", nn_get_result(nrs, [DATA[x][0], DATA[x][1]], false));

        d.draw_text(
            &res,
            abs_x + (DEF_PADDING * 3),
            abs_y + (x as i32 * DEF_PADDING),
            TEXT_DIM,
            Color::WHITE,
        );
    }
}

fn nn_draw_graph(d: &mut RaylibDrawHandle<'_>, costs_vec: &mut Vec<f64>) {

    d.draw_line(
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y,
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        Color::WHITE,
    );
    d.draw_line(
        GRAPHIC_POS.x - DEF_MARGIN,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        GRAPHIC_POS.x + GRAPHIC_DIM.x,
        GRAPHIC_POS.y + GRAPHIC_DIM.y + DEF_MARGIN,
        Color::WHITE,
    );

    let vec_len = costs_vec.len();

    let mut x_prec: i32 = -1;
    let mut y_prec: i32 = -1;

    for n_point in 1..vec_len {
        let x = (((n_point as f64) / (vec_len as f64)) * GRAPHIC_DIM.x as f64) as i32 + GRAPHIC_POS.x;
        let y = (costs_vec[n_point] * GRAPHIC_DIM.y as f64) as i32 + GRAPHIC_POS.y;
        //Se i due cerchi hanno le stesse coordinate evito di disegnarli due volte
        if !((x_prec == x) && (y_prec == y)) {
            d.draw_circle(x, y, 2.0, Color::RED);
        }
        x_prec = x;
        y_prec = y;
    }
}

fn nn_draw_rate_bar(d: &mut RaylibDrawHandle<'_>, rate_circle: &mut IntVec2, is_rate_button_clicked: &mut bool, rate: &mut f64) {
    d.draw_line(
        RATE_BAR_POS.x,
        RATE_BAR_POS.y,
        RATE_BAR_POS.x + RATE_BAR_DIM.x,
        RATE_BAR_POS.y + RATE_BAR_DIM.y,
        Color::WHITE,
    );

    //Vediamo se il cursore è sul cerchio
    if (((d.get_mouse_x() > (rate_circle.x - RATE_BAR_RADIUS as i32)
        && d.get_mouse_x() < (rate_circle.x + RATE_BAR_RADIUS as i32))
        && d.get_mouse_y() > (rate_circle.y - RATE_BAR_RADIUS as i32)
        && d.get_mouse_y() < (rate_circle.y + RATE_BAR_RADIUS as i32))
        || (*is_rate_button_clicked))
        && (d.get_mouse_x() >= RATE_BAR_POS.x
            && d.get_mouse_x() <= RATE_BAR_POS.x + RATE_BAR_DIM.x)
    {
        *is_rate_button_clicked = true;
        if d.is_mouse_button_down(MouseButton::MOUSE_LEFT_BUTTON) {
            rate_circle.x = d.get_mouse_x();
            *rate = ((rate_circle.x - RATE_BAR_H_PADDING) as f64
                / RATE_BAR_DIM.x as f64)
                * MAX_RATE;
        } else {
            *is_rate_button_clicked = false;
        }
    }

    d.draw_circle(rate_circle.x, rate_circle.y, RATE_BAR_RADIUS, Color::PINK);
    
    

}

fn nn_draw_infos(d: &mut RaylibDrawHandle<'_>, rate: f64, cont: u32){

    d.draw_text(
        &format!("Rate: {:.2}", rate),
        TOP_BAR_POS.x,
        TOP_BAR_POS.y,
        TEXT_DIM,
        Color::WHITE,
    );

    d.draw_text(
        &(format!("{} / {}", cont, NUM_TEST)),
        TOP_BAR_POS.x + TOP_BAR_DIM.x,
        TOP_BAR_POS.y + TOP_BAR_DIM.y,
        TEXT_DIM,
        Color::WHITE,
    );

}