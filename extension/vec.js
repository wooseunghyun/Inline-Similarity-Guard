export function l2norm(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  return Math.sqrt(s) + 1e-12;
}

export function normalize(vec) {
  const n = l2norm(vec);
  for (let i = 0; i < vec.length; i++) vec[i] /= n;
  return vec;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
