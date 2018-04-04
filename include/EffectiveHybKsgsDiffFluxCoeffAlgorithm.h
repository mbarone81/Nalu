/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef EffectiveHybKsgsDiffFluxCoeffAlgorithm_h
#define EffectiveHybKsgsDiffFluxCoeffAlgorithm_h

#include<Algorithm.h>

#include<FieldTypeDef.h>

namespace sierra{
namespace nalu{

class Realm;

class EffectiveHybKsgsDiffFluxCoeffAlgorithm : public Algorithm
{
public:

  EffectiveHybKsgsDiffFluxCoeffAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    ScalarFieldType *visc,
    ScalarFieldType *tvisc,
    ScalarFieldType *evisc,
    const double sigmaOne,
    const double sigmaTwo,
    const double sigmaKsgs);
  virtual ~EffectiveHybKsgsDiffFluxCoeffAlgorithm() {}
  virtual void execute();

  ScalarFieldType *visc_;
  ScalarFieldType *tvisc_;
  ScalarFieldType *evisc_;
  ScalarFieldType *fOneBlend_;
  ScalarFieldType *hybBlend_;
  const double sigmaOne_;
  const double sigmaTwo_;
  const double sigmaKsgs_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
