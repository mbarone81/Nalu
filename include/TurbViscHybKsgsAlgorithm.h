/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef TurbViscHybKsgsAlgorithm_h
#define TurbViscHybKsgsAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class TurbViscHybKsgsAlgorithm : public Algorithm
{
public:
  
  TurbViscHybKsgsAlgorithm(
    Realm &realm,
    stk::mesh::Part *part);
  virtual ~TurbViscHybKsgsAlgorithm() {}
  virtual void execute();

  const double aOne_;
  const double betaStar_;
  const double cmuEps_;

  ScalarFieldType *density_;
  ScalarFieldType *viscosity_;
  ScalarFieldType *tke_;
  ScalarFieldType *sdr_;
  ScalarFieldType *minDistance_;
  GenericFieldType *dudx_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *tviscSST_;
  ScalarFieldType *hybridBlending_;
  ScalarFieldType *fLNS_;
  ScalarFieldType *dualNodalVolume_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
