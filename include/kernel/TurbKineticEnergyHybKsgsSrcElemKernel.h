/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/

#ifndef TurbKineticEnergyHybKsgsSrcElemKernel_H
#define TurbKineticEnergyHybKsgsSrcElemKernel_H

#include "kernel/Kernel.h"
#include "FieldTypeDef.h"

#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>

#include <Kokkos_Core.hpp>

namespace sierra {
namespace nalu {

class SolutionOptions;
class MasterElement;
class ElemDataRequests;

/** Add Ksgs source term for kernel-based algorithm approach
 */
template<typename AlgTraits>
class TurbKineticEnergyHybKsgsSrcElemKernel: public Kernel
{
public:
  TurbKineticEnergyHybKsgsSrcElemKernel(
    const stk::mesh::BulkData&,
    const SolutionOptions&,
    ElemDataRequests&);

  virtual ~TurbKineticEnergyHybKsgsSrcElemKernel();

  /** Execute the kernel within a Kokkos loop and populate the LHS and RHS for
   *  the linear solve
   */
  virtual void execute(
    SharedMemView<DoubleType**>&,
    SharedMemView<DoubleType*>&,
    ScratchViews<DoubleType>&);

private:
  TurbKineticEnergyHybKsgsSrcElemKernel() = delete;

  VectorFieldType *coordinates_{nullptr};
  ScalarFieldType *tkeNp1_{nullptr};
  ScalarFieldType *sdrNp1_{nullptr};
  ScalarFieldType *densityNp1_{nullptr};
  ScalarFieldType *tvisc_{nullptr};
  ScalarFieldType *dualNodalVolume_{nullptr};
  GenericFieldType *Gju_{nullptr};

  // Hybrid SST-Ksgs blending function field
  ScalarFieldType *fHatBlend_{nullptr};

  double cEps_{0.0};
  double betaStar_{0.0};
  double tkeProdLimitRatio_{0.0};
  
  /// Integration point to node mapping
  const int* ipNodeMap_;
};

}  // nalu
}  // sierra

#endif /* TurbKineticEnergyHybKsgsSrcElemKernel_H */
