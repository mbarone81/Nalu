/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbKineticEnergyHybKsgsNodeSourceSuppAlg_h
#define TurbKineticEnergyHybKsgsNodeSourceSuppAlg_h

#include <SupplementalAlgorithm.h>
#include <FieldTypeDef.h>

#include <stk_mesh/base/Entity.hpp>

namespace sierra{
namespace nalu{

class Realm;

class TurbKineticEnergyHybKsgsNodeSourceSuppAlg : public SupplementalAlgorithm
{
public:

  TurbKineticEnergyHybKsgsNodeSourceSuppAlg(
    Realm &realm);

  virtual ~TurbKineticEnergyHybKsgsNodeSourceSuppAlg() {}

  virtual void setup();

  virtual void node_execute(
    double *lhs,
    double *rhs,
    stk::mesh::Entity node);
  
  ScalarFieldType *tkeNp1_;
  ScalarFieldType *sdrNp1_;
  ScalarFieldType *densityNp1_;
  ScalarFieldType *tvisc_;
  GenericFieldType *dudx_;
  ScalarFieldType *fHatBlend_;
  ScalarFieldType *dualNodalVolume_;
  double cEps_;
  double betaStar_;
  double tkeProdLimitRatio_;
  int nDim_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
