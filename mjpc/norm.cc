// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/norm.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <mujoco/mujoco.h>

namespace mjpc {

// norm's number of parameters
int NormParameterDimension(int type) {
  switch (type) {
    case NormType::kNull:
      return 0;
    case NormType::kQuadratic:
      return 0;
    case NormType::kL22:
      return 2;
    case NormType::kL2:
      return 1;
    case NormType::kCosh:
      return 1;
    case NormType::kPowerLoss:
      return 1;
    case NormType::kSmoothAbsLoss:
      return 1;
    case NormType::kSmoothAbs2Loss:
      return 2;
    case NormType::kRectifyLoss:
      return 1;
    case NormType::kExpoL2:
      return 3;
    case NormType::kSigmoidL2:
      return 3;
  }
  return 0;
}

// evaluate norm; optionally: gradient, Hessian
double Norm(double* g, double* H, const double* x, const double* params, int n,
            NormType type) {
  if (H && !g) {
    mju_error("Called Norm with H and no g");
  }
  double y = 0;                                                   // output
  double p = params ? params[0] : 0, q = params ? params[1] : 0;  // parameters

  if (H) mju_zero(H, n * n);

  switch (type) {
    case NormType::kNull: {
      y = x[0];
      if (g) {
        g[0] = 1.0;
      }
      if (H) {
        H[0] = 0.0;
      }
      break;
    }

    case NormType::kQuadratic: {  // y = 0.5 * x' * x
      for (int i = 0; i < n; i++) {
        y += x[i] * x[i];
      }
      y *= 0.5;

      if (g) {  // x
        for (int i = 0; i < n; i++) {
          g[i] = x[i];
        }
      }
      if (H) {  // eye(n)
        for (int i = 0; i < n; i++) {
          H[i * n + i] = 1.0;
        }
      }
      break;
    }

    case NormType::kL22: {  // y = ((x*x')^q + p^(2*q))^(1/2/q) - p
      double c = 0;
      for (int i = 0; i < n; i++) {
        c += x[i] * x[i];
      }
      double a = mju_pow(c, q / 2) + mju_pow(p, q);
      double s = mju_pow(a, 1 / q);
      y = s - p;
      double d = mju_pow(c, q / 2 - 1);
      double b = s / a * d;
      if (g) {
        for (int i = 0; i < n; i++) {
          g[i] = b * x[i];
        }
      }

      if (H) {
        c = (1 - q) * d / a + (q - 2) / std::max(c, mjMINVAL);
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            H[i + j * n] = b * ((i == j ? 1.0 : 0.0) + x[i] * x[j] * c);
          }
        }
      }
      break;
    }

    case NormType::kL2: {  // y = sqrt(x*x' + p^2) - p
      double s = mju_sqrt(mju_dot(x, x, n) + p * p);
      y = s - p;
      if (g) {
        if (s) {
          mju_scl(g, x, 1 / s, n);
        } else {
          mju_zero(g, n);
        }
      }

      if (H) {  // H = (eye(n) - g*g')/s
        if (s) {
          for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
              H[i + j * n] = ((i == j ? 1 : 0) - g[i] * g[j]) / s;
            }
          }
        } else {
          mju_zero(H, n * n);
        }
      }
      break;
    }

    case NormType::kCosh: {  // y = p^2 * (cosh(x / p) - 1)
      for (int i = 0; i < n; i++) {
        y += p * p * (std::cosh(x[i] / p) - 1.0);
        if (g) g[i] = p * std::sinh(x[i] / p);
        if (H) H[i * n + i] = std::cosh(x[i] / p);
      }
      break;
    }

    case NormType::kPowerLoss: {  // y = abs(x)^p
      for (int i = 0; i < n; i++) {
        double s = mju_abs(x[i]);
        y += mju_pow(s, p);
        if (g) {
          g[i] = mju_sign(x[i]) * p * mju_pow(s, p - 1);
        }
        if (H) {
          H[i * n + i] = (p - 1) * p * mju_pow(s, p - 2);
        }
      }
      break;
    }

    case NormType::kSmoothAbsLoss: {  // y = sqrt(x^2 + p^2) - p
      for (int i = 0; i < n; i++) {
        double s = mju_sqrt(x[i] * x[i] + p * p);
        y += s - p;
        if (g) g[i] = s ? x[i] / s : 0;
        if (H) H[n * i + i] = s ? (1 - g[i] * g[i]) / s : 0;
      }
      break;
    }

    case NormType::kSmoothAbs2Loss: {  // y = (abs(x)^q + p^q)^(1/q) - p
      for (int i = 0; i < n; i++) {
        double a = mju_abs(x[i]);
        double d = mju_pow(a, q);
        double e = d + mju_pow(p, q);
        double s = mju_pow(e, 1 / q);
        y += s - p;
        double c = s * mju_pow(a, q - 2) / e;
        if (g) g[i] = c * x[i];
        if (H) H[i * n + i] = c * (q - 1) * (1 - d / e);
      }
      break;
    }

    case NormType::kRectifyLoss: {  // y  =  p*log(1 + exp(x/p))
      for (int i = 0; i < n; i++) {
        if (p > 0) {
          double s = mju_exp(x[i] / p);
          y += p * mju_log(1 + s);
          if (g) g[i] = s / (1 + s);
          if (H) H[i * n + i] = s / (p * (1 + s) * (1 + s));
        } else {
          y += x[i] > 0 ? x[i] : 0;
          if (g) g[i] = x[i] > 0 ? 1 : 0;
          if (H) H[i * n + i] = 0;
        }
      }
      break;
    }

    case NormType::kExpoL2: {
      double gamma = params[0];
      double front_factor = params[1];
      double back_factor = params[2];
      double factor = front_factor;
      // Order of the legs based on the residual : FR-FL-RR-RL.
      for (int i = 0; i < 4; i++) {
        factor = (i < 2) ? front_factor : back_factor;
        double s = mju_exp(-gamma * x[3*i+2]);
        double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
        y += factor * s * v;
      }

      if (g) {  // x
        for (int i = 0; i < 4; i++) {
          factor = (i < 2) ? front_factor : back_factor;
          double s = mju_exp(-gamma * x[3*i+2]);
          double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
          g[3*i] = factor * 2*x[3*i] * s;
          g[3*i+1] = factor * 2*x[3*i+1] * s;
          g[3*i+2] = - factor * gamma * s * v;
        }
      }
      if (H) {  // eye(n)
        for (int i = 0; i < 4; i++) {
          factor = (i < 2) ? front_factor : back_factor;
          double s = mju_exp(-gamma * x[3*i+2]);
          double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
          H[(3 * i + 0) * n + 3 * i + 0] = factor * 2 * s;
          H[(3 * i + 0) * n + 3 * i + 1] = 0.;
          H[(3 * i + 0) * n + 3 * i + 2] = - factor * 2. * x[3*i] * gamma * s;

          H[(3 * i + 1) * n + 3 * i + 0] = 0.;
          H[(3 * i + 1) * n + 3 * i + 1] = factor * 2 * s;
          H[(3 * i + 1) * n + 3 * i + 2] = - factor * 2. * x[3*i + 1] * gamma * s;

          H[(3 * i + 2) * n + 3 * i + 0] = - factor * 2. * x[3*i] * gamma * s;
          H[(3 * i + 2) * n + 3 * i + 1] = - factor * 2. * x[3*i + 1] * gamma * s;
          H[(3 * i + 2) * n + 3 * i + 2] = factor * gamma * gamma * s * v;
        }
      }
      break;
    }

    case NormType::kSigmoidL2: {
      double gamma = -params[0];
      double front_factor = params[1];
      double back_factor = params[2];
      double factor = front_factor;
      // Order of the legs based on the residual : FR-FL-RR-RL.
      for (int i = 0; i < 4; i++) {
        factor = (i < 2) ? front_factor : back_factor;
        double e = mju_exp(-gamma * x[3*i+2]);
        double sig = mju_pow(1 + e, -1);
        double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
        // Implementation
        y += factor * sig * v;
      }

      if (g) {  // x
        for (int i = 0; i < 4; i++) {
          factor = (i < 2) ? front_factor : back_factor;
          double e = mju_exp(-gamma * x[3*i+2]);
          double sig = mju_pow(1 + e, -1);
          double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
          // Implementation
          g[3*i] = factor * 2*x[3*i] * sig;
          g[3*i+1] = factor * 2*x[3*i+1] * sig;
          g[3*i+2] = factor * gamma * e * v * mju_pow(1 + e, -2);
        }
      }
      if (H) {  // eye(n)
        for (int i = 0; i < 4; i++) {
          factor = (i < 2) ? front_factor : back_factor;
          double e = mju_exp(-gamma * x[3*i+2]);
          double sig = mju_pow(1 + e, -1);
          double dsig = gamma * e * mju_pow(1 + e, -2);
          double v = x[3*i] * x[3*i] + x[3*i+1] * x[3*i+1];
          // Implementation
          H[(3 * i + 0) * n + 3 * i + 0] = factor * 2 * sig;
          H[(3 * i + 0) * n + 3 * i + 1] = 0.;
          H[(3 * i + 0) * n + 3 * i + 2] = factor * 2. * x[3*i] * dsig;

          H[(3 * i + 1) * n + 3 * i + 0] = 0.;
          H[(3 * i + 1) * n + 3 * i + 1] = factor * 2 * sig;
          H[(3 * i + 1) * n + 3 * i + 2] = factor * 2. * x[3*i + 1] * dsig;

          H[(3 * i + 2) * n + 3 * i + 0] = factor * 2. * x[3*i] * dsig;
          H[(3 * i + 2) * n + 3 * i + 1] = factor * 2. * x[3*i + 1] * dsig;
          H[(3 * i + 2) * n + 3 * i + 2] =  factor * v * (- gamma * dsig + 2 * e * dsig * sig);
        }
      }
      break;
    }

    default:
      mju_error("mj_norm: unknown norm type");
  }
  return y;
}

}  // namespace mjpc
