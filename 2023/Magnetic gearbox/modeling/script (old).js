// Setting constants
const AMOUNT_OF_GEARS = 3
const GEAR_R_RL = getComputedStyle(document.documentElement).getPropertyValue("--gear-r")
const MAGNET_R_RL = getComputedStyle(document.documentElement).getPropertyValue("--magnet-r")
const DIST_BETWEEN_RL = getComputedStyle(document.documentElement).getPropertyValue("--distance-between")
const MAGNET_MASS_RL = 0.05
const ROD_MASS_RL = 0.025
const GEAR_TEMPLATE = document.querySelector("#magnet-template")
const GEARS_ELS = document.querySelectorAll(".gear")

// Setting mathematical functions
const fn_f_from_d = (l) => 0.0000017 / l ** 4
const fn_omega_with_point = (momentum) => momentum[2] / (3 * GEAR_R_RL ** 2 * (MAGNET_MASS_RL + ROD_MASS_RL / 3))
const vec_multiply = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
function get_pp_sec_point(point1, point2, point3) {
  const a_sq = (point2[0] - point3[0]) ** 2 + (point2[1] - point3[1]) ** 2
  const b_sq = (point3[0] - point1[0]) ** 2 + (point3[1] - point1[1]) ** 2
  const dx_c = point2[0] - point1[0]
  const dy_c = point2[1] - point1[1]
  const c_sq = dx_c ** 2 + dy_c ** 2
  const new_x = point2[0] - ((a_sq + c_sq - b_sq) * dx_c) / (2 * c_sq)
  const new_y = point2[1] - ((a_sq + c_sq - b_sq) * dy_c) / (2 * c_sq)
  return [new_x, new_y]
}

// Settings functions
function set_magnets(gear_el, magnets_els, angle = 0) {
  const w = parseFloat(getComputedStyle(gear_el).getPropertyValue("width"))
  const h = parseFloat(getComputedStyle(gear_el).getPropertyValue("height"))
  // console.log(w, h)
  for (let i = 0; i < magnets_els.length; i++) {
    const c_angle = angle + (i * 2 * Math.PI) / magnets_els.length
    const el = magnets_els[i]
    const m_w = parseFloat(getComputedStyle(el).getPropertyValue("width"))
    const m_h = parseFloat(getComputedStyle(el).getPropertyValue("height"))
    const x = (w - m_w + (w - m_w) * Math.cos(c_angle)) / 2
    const y = (h - m_h + (h - m_h) * Math.sin(c_angle)) / 2
    const style = `top: ${y}px; left: ${x}px`
    el.setAttribute("style", style)
  }
}

function set_magnets_rl(gear_center, magnets_list_rl, angle = 0) {
  let c_x = gear_center[0]
  let c_y = gear_center[1]
  for (let i = 0; i < Object.keys(magnets_list_rl).length; i++) {
    const c_angle = angle + (i * 2 * Math.PI) / Object.keys(magnets_list_rl).length
    const x = c_x + GEAR_R_RL * Math.cos(c_angle)
    const y = c_y + GEAR_R_RL * Math.sin(c_angle)
    magnets_list_rl[i][0] = x
    magnets_list_rl[i][1] = y
    // console.log(x, y)
  }
}

function get_momentum(magnets_gear_drive, driven_g_center, magnets_gear_driven) {
  let momentum = [0, 0, 0]
  for (let crd_driven_idx in magnets_gear_driven) {
    let f_r = [0, 0, 0]
    const crd_driven = magnets_gear_driven[crd_driven_idx]
    for (let crd_drive_idx in magnets_gear_drive) {
      const crd_drive = magnets_gear_drive[crd_drive_idx]
      const dx = crd_driven[0] - crd_drive[0]
      const dy = crd_driven[1] - crd_drive[1]
      const l = Math.sqrt(dx ** 2 + dy ** 2)
      // console.log((dx * fn_f_from_d(l)) / l, (dy * fn_f_from_d(l)) / l)
      f_r[0] += (dx * fn_f_from_d(l)) / l
      f_r[1] += (dy * fn_f_from_d(l)) / l
    }
    // console.log(f_r)
    const r_s_p = get_pp_sec_point(crd_driven, [crd_driven[0] + f_r[0], crd_driven[1] + f_r[1]], driven_g_center)
    // console.log(r_s_p)
    const r_vec = [r_s_p[0] - driven_g_center[0], r_s_p[1] - driven_g_center[1], 0]
    // console.log(r_vec)
    const m_a = vec_multiply(r_vec, f_r)
    // console.log(m_a)
    momentum[0] += m_a[0]
    momentum[1] += m_a[1]
    momentum[2] += m_a[2]
  }
  return momentum
}

// Generating gears
let MAGNET_LIST = {}
let MAGNET_LIST_RL = {}
GEARS_ELS.forEach((gear_el) => {
  MAGNET_LIST[gear_el.id] = []
  MAGNET_LIST_RL[gear_el.id] = {}
  for (let i = 0; i < AMOUNT_OF_GEARS; i++) {
    const element = GEAR_TEMPLATE.content.querySelector(".magnet").cloneNode(true)
    element.setAttribute("id", `magnet ${i + 1}`)
    gear_el.appendChild(element)
    MAGNET_LIST[gear_el.id] = MAGNET_LIST[gear_el.id].concat(element)
    MAGNET_LIST_RL[gear_el.id][i] = [0, 0]
  }
  set_magnets(gear_el, MAGNET_LIST[gear_el.id])
})

set_magnets(GEARS_ELS.item(0), MAGNET_LIST["gear-1"])
const g1_c = [-DIST_BETWEEN_RL / 2, 0]
const g2_c = [DIST_BETWEEN_RL / 2, 0]
set_magnets_rl(g1_c, MAGNET_LIST_RL["gear-1"])
set_magnets_rl(g2_c, MAGNET_LIST_RL["gear-2"])
// console.log(MAGNET_LIST_RL)
// console.log(get_momentum(MAGNET_LIST_RL["gear-1"], g2_c, MAGNET_LIST_RL["gear-2"]))

const OMAGE_DRIVE = 0.01
let drive_angle = 0
let omega_driven = 0
let driven_angle = 0
let flag = 1
function calculate() {
  const momentum = get_momentum(MAGNET_LIST_RL["gear-1"], g2_c, MAGNET_LIST_RL["gear-2"])
  const owp = fn_omega_with_point(momentum)
  // console.log(omega_driven)
  drive_angle += OMAGE_DRIVE
  set_magnets(GEARS_ELS.item(0), MAGNET_LIST["gear-1"], drive_angle)
  set_magnets_rl(g1_c, MAGNET_LIST_RL["gear-1"], drive_angle)
  driven_angle += omega_driven + owp / 2
  omega_driven = owp
  set_magnets(GEARS_ELS.item(1), MAGNET_LIST["gear-2"], driven_angle)
  set_magnets_rl(g2_c, MAGNET_LIST_RL["gear-2"], driven_angle)
  if (flag == 1) {
    requestAnimationFrame(calculate)
  }
}
calculate()

// let a = 0
// function mv_forward() {
//   set_magnets(GEARS_ELS[0], MAGNET_LIST["gear-1"], a)
//   set_magnets_rl([0, 0], MAGNET_LIST_RL["gear-1"], a)
//   console.log(MAGNET_LIST_RL)
//   a += 30
//   if (a <= 120) {
//     requestAnimationFrame(mv_forward)
//   }
// }
// mv_forward()
