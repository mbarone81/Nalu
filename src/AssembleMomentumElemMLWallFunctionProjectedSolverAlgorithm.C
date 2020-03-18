/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm.h>
#include <SolverAlgorithm.h>
#include <EquationSystem.h>
#include <LinearSystem.h>
#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

//==========================================================================
// Class Definition
//==========================================================================
// AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm - elem wall function
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm::AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  EquationSystem *eqSystem,
  const bool &useShifted)
  : SolverAlgorithm(realm, part, eqSystem),
    useShifted_(useShifted)
{
  // save off fields
  stk::mesh::MetaData & meta_data = realm_.meta_data();
  exposedAreaVec_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "exposed_area_vector");
  vectorTauWallBip_ = meta_data.get_field<GenericFieldType>(meta_data.side_rank(), "vector_tau_wall_bip");

}

//--------------------------------------------------------------------------
//-------- initialize_connectivity -----------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm::initialize_connectivity()
{

  for ( size_t k = 0; k < partVec_.size(); ++k ) {
    stk::mesh::PartVector partVec;
    partVec.push_back(partVec_[k]);
    eqSystem_->linsys_->buildFaceToNodeGraph(partVec);
  }
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
AssembleMomentumElemMLWallFunctionProjectedSolverAlgorithm::execute()
{
  stk::mesh::MetaData & meta_data = realm_.meta_data();

  const int nDim = meta_data.spatial_dimension();

  // space for LHS/RHS; nodesPerFace*nDim*nodesPerFace*nDim and nodesPerFace*nDim
  std::vector<double> lhs;
  std::vector<double> rhs;
  std::vector<int> scratchIds;
  std::vector<double> scratchVals;
  std::vector<stk::mesh::Entity> connected_nodes;

  // iterate over parts to match construction (requires global counter over locally owned faces)
  for ( size_t pv = 0; pv < partVec_.size(); ++pv ) {
        
    // define selector (per part)
    stk::mesh::Selector s_locally_owned 
      = meta_data.locally_owned_part() &stk::mesh::Selector(*partVec_[pv]);
    
    stk::mesh::BucketVector const& face_buckets =
      realm_.get_buckets( meta_data.side_rank(), s_locally_owned );
    for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
          ib != face_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      
      // face master element
      MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
      const int nodesPerFace = meFC->nodesPerElement_;
      const int numScsBip = meFC->numIntPoints_;
      
      // mapping from ip to nodes for this ordinal; face perspective (use with face_node_relations)
      const int *faceIpNodeMap = meFC->ipNodeMap();

      // resize some things; matrix related
      const int lhsSize = nodesPerFace*nDim*nodesPerFace*nDim;
      const int rhsSize = nodesPerFace*nDim;
      lhs.resize(lhsSize);
      rhs.resize(rhsSize);
      scratchIds.resize(rhsSize);
      scratchVals.resize(rhsSize);
      connected_nodes.resize(nodesPerFace);
      
      // pointers
      double *p_lhs = &lhs[0];
      double *p_rhs = &rhs[0];
      
      const stk::mesh::Bucket::size_type length   = b.size();
      
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        // zero lhs/rhs
        for ( int p = 0; p < lhsSize; ++p )
          p_lhs[p] = 0.0;
        for ( int p = 0; p < rhsSize; ++p )
          p_rhs[p] = 0.0;
        
        // get face
        stk::mesh::Entity face = b[k];
        
        // pointer to face data
        const double *areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
        const double *vectorTauWallBip = stk::mesh::field_data(*vectorTauWallBip_, face);
        
        for ( int ip = 0; ip < numScsBip; ++ip ) {
          
          // offsets
          const int ipNdim = ip*nDim;
          
          const int localFaceNode = faceIpNodeMap[ip];

          // scs area, aMag
          double aMag = 0.0;
          for ( int j = 0; j < nDim; ++j ) {
            const double axj = areaVec[ipNdim+j];
            aMag += axj*axj;
          }
          aMag = std::sqrt(aMag);
          
          // start the rhs assembly (lhs neglected)
          for ( int i = 0; i < nDim; ++i ) {
            
            int indexR = localFaceNode*nDim + i;
            p_rhs[indexR] -= vectorTauWallBip[ipNdim+i]*aMag;
          }
        }
        
        apply_coeff(connected_nodes, scratchIds, scratchVals, rhs, lhs, __FILE__);
        
      }
    }
  }
}

} // namespace nalu
} // namespace Sierra
