window.__pollObservation = function () {
  const sct = slither.sct + slither.rsc; // From game.js
  const out = {
    player: {},
    enemies: {
      xx: [],
      yy: [],
      speed: [],
      heading: [],
      length: [],
      boosting: [],
      segments_x: [],
      segments_y: [],
    },
    food: {
      xx: [],
      yy: [],
      size: [],
    },
    score: 0,
    world_center: 0,
    world_radius: 0,
    minimap: [],
    terminated: false,
  };

  // Death
  if (dead_mtm != -1 || !playing || !connected) {
    out.terminated = true;
    return out;
  } else if (!slither) {
    return out;
  }

  out.score = Math.floor((fpsls[sct] + slither.fam / fmlts[sct] - 1) * 15 - 5);
  out.world_center = grd;
  out.world_radius = real_flux_grd;

  // Snakes
  out.player = {
    xx: slither.xx,
    yy: slither.yy,
    speed: slither.sp || 0,
    heading: slither.ang || 0,
    length: slither.pts?.length || 0,
    boosting: slither.md,
    segments_x: slither.pts.map((s) => s.xx),
    segments_y: slither.pts.map((s) => s.yy),
  };
  for (var i = 0; i < slithers.length; i++) {
    const s = slithers[i];
    // iiv = is in view
    if (!s || !s.alive || !o.iiv || s === slither) continue;

    out.enemies.xx.push(s.xx);
    out.enemies.yy.push(s.yy);
    out.enemies.speed.push(s.speed || 0);
    out.enemies.heading.push(s.heading || 0);
    out.enemies.length.push(s.pts?.length || 0);
    out.enemies.boosting.push(s.md);
    out.enemies.xx.push(s.pts.map((s) => s.xx));
    out.enemies.xx.push(s.pts.map((s) => s.yy));
  }

  // Food filtering bounds
  fpx1 = view_xx - (mww2 / gsc + 24);
  fpy1 = view_yy - (mhh2 / gsc + 24);
  fpx2 = view_xx + (mww2 / gsc + 24);
  fpy2 = view_yy + (mhh2 / gsc + 24);

  // Food
  for (var i = foods_c - 1; i >= 0; i--) {
    const fo = foods[i];
    if (
      !fo.eaten &&
      fo.rx >= fpx1 &&
      fo.ry >= fpy1 &&
      fo.rx <= fpx2 &&
      fo.ry <= fpy2
    ) {
      out.food.xx.push(fo.xx);
      out.food.yy.push(fo.yy);
      out.food.size.push(fo.sz);
    }
  }
  for (var i = preys.length - 1; i >= 0; i--) {
    const pr = preys[i];
    if (
      !pr.eaten &&
      pr.rx >= fpx1 &&
      pr.ry >= fpy1 &&
      pr.rx <= fpx2 &&
      pr.ry <= fpy2
    ) {
      out.food.xx.push(pr.xx);
      out.food.yy.push(pr.yy);
      out.food.size.push(pr.sz);
    }
  }

  // Minimap
  const normalising_constant = (mmsz - 1) / 2; // Normalize to [0, 2]
  const centering_bias = 1; // Bias to [-1, 1]
  for (var i = mmdata.length - 1; i >= 0; i--) {
    if (mmdata[i] === 0) continue;

    // Normalize and center the coordinates around the map center.
    out.minimap.push({
      x: (i % mmsz) / normalising_constant - centering_bias,
      y: Math.floor(i / mmsz) / normalising_constant - centering_bias,
    });
  }
  return out;
};
