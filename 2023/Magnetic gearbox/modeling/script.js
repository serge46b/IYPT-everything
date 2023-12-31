// ---Setting constants---
// real life parameters
const MAGNETS_AMOUNT = 3
const GEAR_R = 0.041
const MAGNET_R = 0.01
const DIST_BETWEEN = 0.1
const MAGNET_MASS = 0.05
const BRANCH_MASS = 0.005
const T_K = 0.2
// programm constants
const GEAR_TEMPLATE = document.querySelector("#gear-template")
const MAGNET_TEMPLATE = document.querySelector("#magnet-template")
const CANVAS_EL = document.querySelector("#canvas")
const CONTAINER_EL = document.querySelector("body")
const svgns = "http://www.w3.org/2000/svg"
const SOUTH = 1
const NORTH = -1
// -----------------------

// Setting mathematical functions
const fn_f_from_d = (l) => 0.0000017 / l ** 4
const fn_omega_with_point = (moment) => moment[2] / (MAGNETS_AMOUNT * GEAR_R ** 2 * (MAGNET_MASS + BRANCH_MASS / 3))
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

function calc_forces_and_get_moment(gears_list) {
  let moment = [0, 0, 0]
  gears_list[1].magnets.forEach((driven_magnet) => {
    let resulting_froce = [0, 0, 0]
    const crd_dvn = driven_magnet.coordinates
    gears_list[0].magnets.forEach((drive_magnet, mg_idx) => {
      const crd_dve = drive_magnet.coordinates
      const dx = crd_dvn[0] - crd_dve[0]
      const dy = crd_dvn[1] - crd_dve[1]
      const l = Math.sqrt(dx ** 2 + dy ** 2)
      const calc_force = [
        (dx * fn_f_from_d(l) * drive_magnet.pole) / l,
        (dy * fn_f_from_d(l) * drive_magnet.pole) / l,
        0,
      ]
      driven_magnet.forces.magnetic[mg_idx] = calc_force
      resulting_froce[0] += calc_force[0]
      resulting_froce[1] += calc_force[1]
    })
    const rsp = get_pp_sec_point(
      crd_dvn,
      [crd_dvn[0] + resulting_froce[0], crd_dvn[1] + resulting_froce[1]],
      gears_list[1].center
    )
    const r_vec = [rsp[0] - gears_list[1].center[0], rsp[1] - gears_list[1].center[1], 0]
    const m_a = vec_multiply(r_vec, resulting_froce)
    moment[0] += m_a[0]
    moment[1] += m_a[1]
    moment[2] += m_a[2]
  })
  return moment
}

// Settings functions
function set_magnets(gear, angle = 0) {
  let c_x = gear.center[0]
  let c_y = gear.center[1]
  for (let i = 0; i < gear.magnets.length; i++) {
    const c_angle = angle + (i * 2 * Math.PI) / gear.magnets.length
    const x = c_x + GEAR_R * Math.cos(c_angle)
    const y = c_y + GEAR_R * Math.sin(c_angle)
    gear.magnets[i].coordinates[0] = x
    gear.magnets[i].coordinates[1] = y
    // console.log(x, y)
  }
}

// Rendering
function render(gears_list, canvas_el) {
  gears_list.forEach((gear, g_idx) => {
    const gear_c_x = gear.center[0]
    const gear_c_y = gear.center[1]
    const gear_group = canvas_el.querySelectorAll(".gear").item(g_idx)
    gear.magnets.forEach((magnet, mg_idx) => {
      const teeth_group_el = gear_group.querySelectorAll(".teeth").item(mg_idx)
      const circle_el = teeth_group_el.querySelector(".magnet")
      const branch_el = teeth_group_el.querySelector(".branch")
      circle_el.setAttribute("cx", magnet.coordinates[0])
      circle_el.setAttribute("cy", magnet.coordinates[1])
      branch_el.setAttribute("x1", magnet.coordinates[0])
      branch_el.setAttribute("y1", magnet.coordinates[1])
      branch_el.setAttribute("x2", gear_c_x)
      branch_el.setAttribute("y2", gear_c_y)
    })
  })
}

// Generating gears
let gears_list = [
  { center: [-DIST_BETWEEN / 2, 0], magnets: [] },
  { center: [DIST_BETWEEN / 2, 0], magnets: [] },
]
gears_list.forEach((gear) => {
  for (let i = 0; i < MAGNETS_AMOUNT; i++) {
    gear.magnets[i] = { coordinates: [0, 0], pole: SOUTH, forces: { magnetic: [] } }
    // ----TMP----
    // gear.magnets[i].pole = i == 0 ? SOUTH : NORTH
    // -----------
    if (gear.magnets[i].pole < -1 || gear.magnets[i].pole > 1) {
      gear.magnets[i].pole = SOUTH
    }
    for (let j = 0; j < MAGNETS_AMOUNT; j++) {
      gear.magnets[i].forces.magnetic[j] = 0
    }
  }
  set_magnets(gear)
})

// Canvas preparations
const cnv_height = (GEAR_R + MAGNET_R) * 2
const cnv_width = (GEAR_R + MAGNET_R) * 2 + DIST_BETWEEN
CANVAS_EL.setAttribute("viewBox", `${-cnv_width / 2} ${-cnv_height / 2} ${cnv_width} ${cnv_height}`)
gears_list.forEach((gear, g_idx) => {
  const gear_group_el = GEAR_TEMPLATE.content.querySelector(".gear").cloneNode(true)
  gear_group_el.setAttribute("id", `gear-${g_idx + 1}`)
  gear.magnets.forEach((magnet, mg_idx) => {
    const magnet_group_el = MAGNET_TEMPLATE.content.querySelector(".teeth").cloneNode(true)
    magnet_group_el.classList.add(`teeth-${mg_idx + 1}`)
    const circle_el = magnet_group_el.querySelector(".magnet")
    circle_el.setAttribute("r", MAGNET_R)
    gear_group_el.appendChild(magnet_group_el)
  })
  CANVAS_EL.appendChild(gear_group_el)
})
render(gears_list, CANVAS_EL)

const OMAGE_DRIVE = 2
let drive_angle = 0
let omega_driven = 0
let driven_angle = 0
let l_t = 0
let flag = 1
function calculate() {
  const moment = calc_forces_and_get_moment(gears_list)
  const owp = fn_omega_with_point(moment)
  // console.log(omega_driven)
  const dt = ((Date.now() - l_t) * T_K) / 1000
  drive_angle += OMAGE_DRIVE * dt
  set_magnets(gears_list[0], drive_angle)
  driven_angle += omega_driven * dt + (owp * dt * dt) / 2
  omega_driven = owp * dt
  set_magnets(gears_list[1], driven_angle)
  render(gears_list, CANVAS_EL)
  if (flag == 1) {
    l_t = Date.now()
    requestAnimationFrame(calculate)
  }
}
function start_clac() {
  l_t = Date.now()
  requestAnimationFrame(calculate)
}
start_clac()

// let a = 0
// function mv_forward() {
//   set_magnets(GEARS_ELS[0], MAGNET_LIST["gear-1"], a)
//   set_magnets([0, 0], MAGNET_LIST["gear-1"], a)
//   console.log(MAGNET_LIST)
//   a += 30
//   if (a <= 120) {
//     requestAnimationFrame(mv_forward)
//   }
// }
// mv_forward()
