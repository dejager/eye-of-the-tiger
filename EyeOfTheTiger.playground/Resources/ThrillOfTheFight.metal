#include <metal_stdlib>
using namespace metal;

#define uvScale 1.0
#define colorUvScale 0.1
#define furDepth 0.2
#define furLayers 128.0
#define rayStep furDepth * 2.0 / furLayers
#define furThreshold 0.4
#define shininess 50.0

float sphere(float3 o, float3 d, float r, float t) {
  float b = dot(-o, d);
  float det = b * b - dot(o, o) + r * r;
  if (det < 0.0) return false;
  det = sqrt(det);
  t = b - det;
  return t;
}


float3 rotateX(float3 p, float a) {
  float sinA = sin(a);
  float cosA = cos(a);
  return float3(p.x, cosA * p.y - sinA * p.z, sinA * p.y + cosA * p.z);
}

float3 rotateY(float3 p, float a) {
  float sinA = sin(a);
  float cosA = cos(a);
  return float3(cosA * p.x + sinA * p.z, p.y, -sinA * p.x + cosA * p.z);
}

float2 cartesianToSpherical(float3 p, float time) {
  float r = length(p);
  float t = (r - (1.0 - furDepth)) / furDepth;

  // add a slight curl
  p = rotateX(p.zyx, -cos(time * 1.5) * t * t * 0.4).zyx;

  p = p / r;
  float2 uv = float2(atan(p.x / p.y), acos(p.z));
  uv.y -= t * t * 0.1;
  return uv;
}

// returns fur density at given position
float density(float3 pos, float2 uv, float time, texture2d<float, access::sample> texture, sampler smp)
{
  //uv = cartesianToSpherical(pos.xzy, time);

  float2 scaledUV = uv * uvScale;
  float4 color = texture.sample(smp, scaledUV);

  // thin out hair
  float density = smoothstep(furThreshold, 1.0, color.x);

  float r = length(pos);
  float t = (r - (1.0 - furDepth)) / furDepth;

  // fade out along length
  float len = color.y;
  density *= smoothstep(len, len-0.2, t);

  return density;
}

// make normal from density
float3 normal(float3 pos, float thickness, float time, texture2d<float, access::sample> texture, sampler smp) {
  float epsilon = 0.01;
  float3 n;
  float2 uv;

  float3 ps = float3(pos.x + epsilon, pos.y, pos.z);
  uv = cartesianToSpherical(ps.xzy, time);
  n.x = density( ps, uv, time, texture, smp) - thickness;

  ps = float3(pos.x, pos.y + epsilon, pos.z);
  uv = cartesianToSpherical(ps.xzy, time);
  n.y = density( ps, uv, time, texture, smp) - thickness;

  ps = float3(pos.x, pos.y, pos.z + epsilon);
  uv = cartesianToSpherical(ps.xzy, time);
  n.z = density( ps, uv, time, texture, smp) - thickness;
  return normalize(n);
}

float3 furShade(float3 pos, float2 uv, float3 origin, float density, float time, texture2d<float, access::sample> texture, texture2d<float, access::sample> colorTexture, sampler smp)
{
  // lights
  const float3 len = float3(1, 1, 0);
  float3 vert = normalize(origin - pos);
  float3 horz = normalize(vert + len);

  float3 norm = -normal(pos, density, time, texture, smp);
  float diff = max(0.0, dot(norm, len) * 0.5 + 0.5);
  float spec = pow(max(0.0, dot(norm, horz)), shininess) * 0.5;

  // starting color
  float2 scaledUV = uv * colorUvScale;
  float3 color = colorTexture.sample(smp, scaledUV).rgb;

  // depth
  float r = length(pos);
  float t = (r - (1.0 - furDepth)) / furDepth;
  t = clamp(t, 0.0, 1.0);
  float i = t * 0.5 + 0.5;

  // leopard
  //spec *= 0.5;
  //return color * diff * i + float3(spec * i, spec * i, 0);

  // sully
  return color * diff * i + float3(0, spec * i, spec * i);
}

float4 environment(float3 ro,
             float3 rd,
             float time,
             texture2d<float, access::sample> texture,
             texture2d<float, access::sample> colorTexture, sampler smp) {

  float3 p = float3(0.0);
  const float r = 1.0;
  float t = sphere(ro - p, rd, r, t);
  bool isInside = t > 0.0;

  float4 c = float4(0.0);
  if (isInside) {
    float3 pos = ro + rd * t;

    for(int i=0; i<furLayers; i++) {
      float4 colorRepresentation;
      float2 uv = cartesianToSpherical(pos.xzy, time);

      colorRepresentation.a = density(pos, uv, time, texture, smp);
      if (colorRepresentation.a > 0.0) {
        colorRepresentation.rgb = furShade(pos, uv, ro, colorRepresentation.a, time, texture, colorTexture, smp);

        // premultiply alpha
        colorRepresentation.rgb *= colorRepresentation.a;
        c = c + colorRepresentation * (1.0 - c.a);
        if (c.a > 0.95) {
          break;
        }
      }
      pos += rd * rayStep;
    }
  }

  return c;
}

kernel void risinUpStraightToTheTop(texture2d<float, access::write> o[[texture(0)]],
                                    texture2d<float, access::sample> input[[texture(1)]],
                                    texture2d<float, access::sample> paint[[texture(2)]],
                                    sampler smp [[ sampler(0) ]],
                                    constant float &time [[buffer(0)]],
                                    constant float2 *touchEvent [[buffer(1)]],
                                    constant int &numberOfTouches [[buffer(2)]],
                                    ushort2 gid [[thread_position_in_grid]]) {

  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 p = float2(gid.xy);

  float2 uv = p / res;
  uv = uv * 2.0 - 1.0;
  uv.x *= res.x / res.y;

  float3 origin = float3(0.0, 0.0, 2.5);
  float3 direction = normalize(float3(uv, -1.5));

  float rotationX = sin(time * 1.5);
  float rotationY = 0.0;

  origin = rotateY(origin, rotationX);
  origin = rotateY(origin, rotationY);
  direction = rotateY(direction, rotationX);
  direction = rotateY(direction, rotationY);

  float4 color = environment(origin, direction, time, input, paint, smp);
  o.write(color, gid);
}
