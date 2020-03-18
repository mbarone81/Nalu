/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm_h
#define AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm_h

#include<SolverAlgorithm.h>
#include<FieldTypeDef.h>

namespace stk {
namespace mesh {
  class Part;
  class Ghosting;
}
}

namespace sierra{
namespace nalu{

class Realm;
class PointInfo;

class AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm : public SolverAlgorithm
{
public:

  AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm(
    Realm &realm,
    stk::mesh::Part *part,
    EquationSystem *eqSystem,
    const bool &useShifted);
  virtual ~AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm() {}
  virtual void initialize_connectivity();
  virtual void execute();

  const bool useShifted_;

  GenericFieldType *exposedAreaVec_;
  GenericFieldType *vectorTauWallBip_;
  
};

} // namespace nalu
} // namespace Sierra

#endif
