/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2009-2026 Stanford University and the Authors.      *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "HipHipFFT3D.h"
#include "HipContext.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

HipHipFFT3D::HipHipFFT3D(HipContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
        context(context), realToComplex(realToComplex), doublePrecision(context.getUseDoublePrecision())
#ifdef OPENMM_HIP_WITH_HIPFFT
        , forwardPlan(0), backwardPlan(0)
#endif
{
#ifndef OPENMM_HIP_WITH_HIPFFT
    throw OpenMMException("hipFFT backend requested but OpenMM was built without HIPFFT support");
#else
    hipfftType forwardType, backwardType;
    if (realToComplex) {
        forwardType = doublePrecision ? HIPFFT_D2Z : HIPFFT_R2C;
        backwardType = doublePrecision ? HIPFFT_Z2D : HIPFFT_C2R;
    }
    else {
        forwardType = doublePrecision ? HIPFFT_Z2Z : HIPFFT_C2C;
        backwardType = forwardType;
    }
    hipfftResult result = hipfftPlan3d(&forwardPlan, xsize, ysize, zsize, forwardType);
    if (result != HIPFFT_SUCCESS)
        throw OpenMMException("Error initializing hipFFT forward plan: "+context.intToString(result));
    result = hipfftPlan3d(&backwardPlan, xsize, ysize, zsize, backwardType);
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(forwardPlan);
        throw OpenMMException("Error initializing hipFFT backward plan: "+context.intToString(result));
    }
#endif
}

HipHipFFT3D::~HipHipFFT3D() {
#ifdef OPENMM_HIP_WITH_HIPFFT
    if (forwardPlan != 0)
        hipfftDestroy(forwardPlan);
    if (backwardPlan != 0)
        hipfftDestroy(backwardPlan);
#endif
}

void HipHipFFT3D::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
#ifndef OPENMM_HIP_WITH_HIPFFT
    throw OpenMMException("hipFFT backend requested but OpenMM was built without HIPFFT support");
#else
    hipfftHandle plan = forward ? forwardPlan : backwardPlan;
    hipfftResult result = hipfftSetStream(plan, context.getCurrentStream());
    if (result != HIPFFT_SUCCESS)
        throw OpenMMException("Error setting hipFFT stream: "+context.intToString(result));
    if (realToComplex) {
        if (forward) {
            if (doublePrecision)
                result = hipfftExecD2Z(plan, (double*) context.unwrap(in).getDevicePointer(), (double2*) context.unwrap(out).getDevicePointer());
            else
                result = hipfftExecR2C(plan, (float*) context.unwrap(in).getDevicePointer(), (float2*) context.unwrap(out).getDevicePointer());
        }
        else {
            if (doublePrecision)
                result = hipfftExecZ2D(plan, (double2*) context.unwrap(in).getDevicePointer(), (double*) context.unwrap(out).getDevicePointer());
            else
                result = hipfftExecC2R(plan, (float2*) context.unwrap(in).getDevicePointer(), (float*) context.unwrap(out).getDevicePointer());
        }
    }
    else {
        if (doublePrecision)
            result = hipfftExecZ2Z(plan, (double2*) context.unwrap(in).getDevicePointer(), (double2*) context.unwrap(out).getDevicePointer(), forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD);
        else
            result = hipfftExecC2C(plan, (float2*) context.unwrap(in).getDevicePointer(), (float2*) context.unwrap(out).getDevicePointer(), forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD);
    }
    if (result != HIPFFT_SUCCESS)
        throw OpenMMException("Error executing hipFFT: "+context.intToString(result));
#endif
}
