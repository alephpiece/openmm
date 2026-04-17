#ifndef __OPENMM_HIPHIPFFT3D_H__
#define __OPENMM_HIPHIPFFT3D_H__

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

#include "openmm/common/windowsExportCommon.h"
#include "openmm/common/FFT3D.h"
#include "openmm/common/ArrayInterface.h"

#ifdef OPENMM_HIP_WITH_HIPFFT
#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif
#endif

namespace OpenMM {

class HipContext;

class OPENMM_EXPORT_COMMON HipHipFFT3D : public FFT3DImpl {
public:
    HipHipFFT3D(HipContext& context, int xsize, int ysize, int zsize, bool realToComplex);
    ~HipHipFFT3D();
    void execFFT(ArrayInterface& in, ArrayInterface& out, bool forward=true) override;
private:
    HipContext& context;
    bool realToComplex;
    bool doublePrecision;
#ifdef OPENMM_HIP_WITH_HIPFFT
    hipfftHandle forwardPlan;
    hipfftHandle backwardPlan;
#endif
};

} // namespace OpenMM

#endif // __OPENMM_HIPHIPFFT3D_H__
