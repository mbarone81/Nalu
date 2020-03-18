/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <ComputeMLTauWallProjectedAlgorithm.h>
#include <Algorithm.h>
#include <PointInfo.h>

#include <FieldTypeDef.h>
#include <Realm.h>
#include <master_element/MasterElement.h>
#include <NaluEnv.h>

#include <utils/StkHelpers.h>

// stk_mesh/base/fem
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_mesh/base/ExodusTranslator.hpp>

// stk_util
#include <stk_util/parallel/ParallelReduce.hpp>

// basic c++
#include <cmath>

namespace sierra{
namespace nalu{

// compare operator - move this....
struct compareId {
  bool operator () (const std::pair<uint64IdentProc, uint64IdentProc> &p, const uint64_t i) {
    return (p.first.id() < i);
  }
  bool operator () (const uint64_t i, const std::pair<uint64IdentProc, uint64IdentProc> &p) {
    return (i < p.first.id());
  }
};

struct lessThan
{
  bool operator() (const std::pair<uint64IdentProc, uint64IdentProc> &p, const std::pair<uint64IdentProc, uint64IdentProc> &q) {
    return (p.first.id() < q.first.id());
  }
};

//==========================================================================
// Class Definition
//==========================================================================
// ComputeMLTauWallProjectedAlgorithm - utau at wall bc
//==========================================================================
//--------------------------------------------------------------------------
//-------- constructor -----------------------------------------------------
//--------------------------------------------------------------------------
ComputeMLTauWallProjectedAlgorithm::ComputeMLTauWallProjectedAlgorithm(
  Realm &realm,
  stk::mesh::Part *part,
  const double projectedDistance,
  const bool useShifted,
  std::vector<std::vector<PointInfo *> > &pointInfoVec,
  stk::mesh::Ghosting *wallFunctionGhosting)
  : Algorithm(realm, part),
    useShifted_(useShifted),
    pointInfoVec_(pointInfoVec),
    wallFunctionGhosting_(wallFunctionGhosting),
    bulkData_(&realm.bulk_data()),
    metaData_(&realm.meta_data()),
    nDim_(realm.meta_data().spatial_dimension()),
    yplusCrit_(11.63),
    elog_(8.432), //elog_(9.8),
    kappa_(realm.get_turb_model_constant(TM_kappa)),
    maxIteration_(20),
    tolerance_(1.0e-6),
    firstInitialization_(true),
    provideOutput_(false),
    searchMethod_(stk::search::KDTREE),
    expandBoxPercentage_(0.05),
    needToGhostCount_(0)
{
  // save off fields
  velocity_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "velocity");
  dudx_ = metaData_->get_field<GenericFieldType>(stk::topology::NODE_RANK, "dudx");
  //dpdx_ = metaData_->get_field<GenericFieldType>(stk::topology::NODE_RANK, "dpdx");
  bcVelocity_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, "wall_velocity_bc");
  coordinates_ = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  density_ = metaData_->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "density");
  viscosity_ = metaData_->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "viscosity");
  exposedAreaVec_ = metaData_->get_field<GenericFieldType>(metaData_->side_rank(), "exposed_area_vector");
  vectorTauWallBip_ = metaData_->get_field<GenericFieldType>(metaData_->side_rank(), "vector_tau_wall_bip");
  wallFrictionVelocityBip_ = metaData_->get_field<GenericFieldType>(metaData_->side_rank(), "wall_friction_velocity_bip");
  wallNormalDistanceBip_ = metaData_->get_field<GenericFieldType>(metaData_->side_rank(), "wall_normal_distance_bip");
  assembledWallArea_ = metaData_->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_area_wf");
  assembledWallNormalDistance_ = metaData_->get_field<ScalarFieldType>(stk::topology::NODE_RANK, "assembled_wall_normal_distance");
  tauLogBip_ = metaData_->get_field<GenericFieldType>(metaData_->side_rank(), "tau_log_bip");
  
  // set data
  set_data(projectedDistance);

  // what do we need ghosted for this alg to work?
  ghostFieldVec_.push_back(&(velocity_->field_of_state(stk::mesh::StateNP1)));
  //ghostFieldVec_.push_back(&(dudx_->field_of_state(stk::mesh::StateNP1)));
  ghostFieldVec_.push_back(dudx_);
  //ghostFieldVec_.push_back(&(dpdx_->field_of_state(stk::mesh::StateNP1)));

  // Neural network setup

  // Read the neural network matrix operators from file
  std::ifstream weightsFile;
  weightsFile.open(weightsFileName_, std::ios_base::in);
  weightsFile >> numNetworks_;
  weights_.resize(numNetworks_);
  weightsFile >> numLayers_;
  numLayRow_.resize(numLayers_);
  numLayCol_.resize(numLayers_);
  for (int i=0; i<numNetworks_; ++i) {
    weights_[i].resize(numLayers_);
    for (int j=0; j<numLayers_; ++j) {
      weightsFile >> numLayRow_[j] >> numLayCol_[j];
      weights_[i][j].resize(numLayRow_[j]*numLayCol_[j]);
      int matSize = numLayRow_[j]*numLayCol_[j];
      for (int k=0; k<matSize; ++k) {
        weightsFile >> weights_[i][j][k];
      }
    }
  }
  // Read in the feature and label scale factors
  // For lambda's, read in min and max values to be used with minmax scaling fcn
  for (int i=0; i<numLambda_; ++i) {
    weightsFile >> fmin_[i] >> fmax_[i];
  }
  // For Pi vectors, read in scalar scale factors
  for (int i=0; i<numPi_; ++i) {
    weightsFile >> sf_Pi_[i];
  }
  // For tau vector labels, read in scalar scale factor
  weightsFile >> sf_tau_;
  weightsFile.close();

  NaluEnv::self().naluOutputP0() << "ML projected wall function in use" << std::endl;
}
  
//--------------------------------------------------------------------------
//-------- destructor ------------------------------------------------------
//--------------------------------------------------------------------------
ComputeMLTauWallProjectedAlgorithm::~ComputeMLTauWallProjectedAlgorithm()
{
  // does nothing
}

//--------------------------------------------------------------------------
//-------- execute ---------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::execute()
{
  // fixed size
  //std::vector<double> uProjected(nDim_);
  std::vector<double> uBcBip(nDim_);
  std::vector<double> unitNormal(nDim_);
  //std::vector<double> cProjected(nDim_);
  //std::vector<double> dudxProjected(nDim_*nDim_);
  //std::vector<double> dpdxProjected(nDim_);
  std::vector<double> uBip(nDim_);
  std::vector<double> dudxBip(nDim_*nDim_);
  
  // pointers to fixed values
  //double *p_uProjected = &uProjected[0];
  double *p_uBcBip = &uBcBip[0];
  double *p_unitNormal= &unitNormal[0];
  double *p_uBip = &uBip[0];
  double *p_dudxBip = &dudxBip[0];
  //double *p_dudxProjected = &dudxProjected[0];
  //double *p_dpdxProjected = &dpdxProjected[0];
  
  // isopar coordinates for the owning element
  std::vector<double> isoParCoords(nDim_);
  
  // nodal fields to gather
  std::vector<double> ws_velocityNp1;
  std::vector<double> ws_bcVelocity;
  std::vector<double> ws_density;
  std::vector<double> ws_viscosity;
  std::vector<double> ws_dudx;
  
  // master element
  std::vector<double> ws_shape_function;
  std::vector<double> ws_face_shape_function;
  
  // deal with state
  VectorFieldType &velocityNp1 = velocity_->field_of_state(stk::mesh::StateNP1);
  ScalarFieldType &densityNp1 = density_->field_of_state(stk::mesh::StateNP1);
  //GenericFieldType &dpdxNp1 = dpdx_->field_of_state(stk::mesh::StateNP1);
  GenericFieldType &dudxNp1 = *dudx_; //->field_of_state(stk::mesh::StateNP1);
  
  // define vector of parent topos; should always be UNITY in size
  std::vector<stk::topology> parentTopo;

  // simple for now...
  initialize();
  
  // parallel communicate ghosted entities
  if ( nullptr != wallFunctionGhosting_ )
    stk::mesh::communicate_field_data(*(wallFunctionGhosting_), ghostFieldVec_);
  
  // iterate over parts to match construction (requires global counter over locally owned faces)
  size_t pointInfoVecCounter = 0;  
  for ( size_t pv = 0; pv < partVec_.size(); ++pv ) {

    // extract projected distance
    //const double pDistance = projectedDistanceVec_[pv];

    // define selector (per part)
    stk::mesh::Selector s_locally_owned 
      = metaData_->locally_owned_part() &stk::mesh::Selector(*partVec_[pv]);
    
    stk::mesh::BucketVector const& face_buckets =
      realm_.get_buckets( metaData_->side_rank(), s_locally_owned );
    
    for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
          ib != face_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;

      // extract connected element topology
      b.parent_topology(stk::topology::ELEMENT_RANK, parentTopo);
      ThrowAssert ( parentTopo.size() == 1 );
      stk::topology theElemTopo = parentTopo[0];
      
      // extract master element
      MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(theElemTopo);

      // face master element
      MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
      const int nodesPerFace = b.topology().num_nodes();
      const int numScsBip = meFC->numIntPoints_;
      
      // mapping from ip to nodes for this ordinal; face perspective (use with face_node_relations)
      const int *faceIpNodeMap = meFC->ipNodeMap();

      // algorithm related; element
      ws_velocityNp1.resize(nodesPerFace*nDim_);
      ws_bcVelocity.resize(nodesPerFace*nDim_);
      ws_density.resize(nodesPerFace);
      ws_viscosity.resize(nodesPerFace);
      ws_face_shape_function.resize(numScsBip*nodesPerFace);
      ws_dudx.resize(nodesPerFace*nDim_*nDim_);
      
      // pointers
      double *p_velocityNp1 = &ws_velocityNp1[0];
      double *p_bcVelocity = &ws_bcVelocity[0];
      double *p_density = &ws_density[0];
      double *p_viscosity = &ws_viscosity[0];
      double *p_face_shape_function = &ws_face_shape_function[0];
      double *p_dudx = &ws_dudx[0];
      
      // shape functions
      if ( useShifted_ )
        meFC->shifted_shape_fcn(&p_face_shape_function[0]);
      else
        meFC->shape_fcn(&p_face_shape_function[0]);

      const stk::mesh::Bucket::size_type length   = b.size();
      
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        // get face
        stk::mesh::Entity face = b[k];
        
        //======================================
        // gather nodal data off of face
        //======================================
        stk::mesh::Entity const * face_node_rels = bulkData_->begin_nodes(face);
        int num_face_nodes = bulkData_->num_nodes(face);
        // sanity check on num nodes
        ThrowAssert( num_face_nodes == nodesPerFace );
        for ( int ni = 0; ni < num_face_nodes; ++ni ) {
          stk::mesh::Entity node = face_node_rels[ni];
          
          // gather scalars
          p_density[ni]    = *stk::mesh::field_data(densityNp1, node);
          p_viscosity[ni] = *stk::mesh::field_data(*viscosity_, node);
          
          // gather vectors
          double * uNp1 = stk::mesh::field_data(velocityNp1, node);
          double * uBc = stk::mesh::field_data(*bcVelocity_, node);
          const int niNdim = ni*nDim_;
          for ( int j=0; j < nDim_; ++j ) {
            p_velocityNp1[niNdim+j] = uNp1[j];
            p_bcVelocity[niNdim+j] = uBc[j];
          }

          // gather tensors
          double *dudx = stk::mesh::field_data(*dudx_, node);
          const int offSetT = ni*nDim_*nDim_;
          for ( int j=0; j < nDim_*nDim_; ++j ) {
            p_dudx[offSetT+j] = dudx[j];
          }
        }
        
        /*std::vector<double> vel_tmp(nDim_*num_face_nodes);
        std::vector<double> dudx_tmp(nDim_*nDim_*num_face_nodes);
        vel_tmp = {0.546219, -0.000134405, -0.0393282, 0.545717, 9.51936e-05, -0.0397933, 0.567866, 0.00149822, -0.0203802, 0.562784, 0.00182906, -0.0175848};
        dudx_tmp = {-0.00692822, 1.62365, 0.137313, 0.00214833, 0.0358497, 0.00149697, -0.0163845, -0.0783211, -0.00352985, 0.00914333, 1.61081, 0.150677, 0.00248075, 0.0461784, -0.00115031, 0.0177873, -0.0824453, -0.0317802, 0.0265501, 1.60583, 0.428842, -0.000696891, 0.0428329, -0.00596769, 0.00348447, -0.0979581, -0.123049, 0.00964764, 1.61369, 0.410387, -0.0017055, 0.0426081, 3.49945e-05, -0.0267705, -0.0835014, -0.109812};
        for (int j=0; j<nDim_*num_face_nodes; ++j) {
          p_velocityNp1[j] = vel_tmp[j];
        }
        for (int j=0; j<nDim_*nDim_*num_face_nodes; ++j) {
          p_dudx[j] = dudx_tmp[j];
        }
        */
        
        // pointer to face data
        const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
        double *wallNormalDistanceBip = stk::mesh::field_data(*wallNormalDistanceBip_, face);
        double *wallFrictionVelocityBip = stk::mesh::field_data(*wallFrictionVelocityBip_, face);
        double *vectorTauWallBip = stk::mesh::field_data(*vectorTauWallBip_, face);
        double *tauLogBip = stk::mesh::field_data(*tauLogBip_, face);
        
        // extract the vector of PointInfo for this face 
        std::vector<PointInfo *> &faceInfoVec = pointInfoVec_[pointInfoVecCounter++];

        // extract the connected element to this exposed face; should be single in size!
        const stk::mesh::Entity* face_elem_rels = bulkData_->begin_elements(face);
        ThrowAssert( bulkData_->num_elements(face) == 1 );

        // get element; its face ordinal number
        stk::mesh::Entity element = face_elem_rels[0];
        const int face_ordinal = bulkData_->begin_element_ordinals(face)[0];

        // get the relations off of element
        stk::mesh::Entity const * elem_node_rels = bulkData_->begin_nodes(element);

        // loop over ips
        for ( int ip = 0; ip < numScsBip; ++ip ) {

          const int ipNdim = ip*nDim_;

          const int opposingNode = meSCS->opposingNodes(face_ordinal,ip);

          // extract nearest node to this ip
          const int localFaceNode = faceIpNodeMap[ip]; 
          stk::mesh::Entity nearestNode = face_node_rels[localFaceNode];

          // left and right nodes; right is on the face; left is the opposing node
          stk::mesh::Entity nodeL = elem_node_rels[opposingNode];
          stk::mesh::Entity nodeR = face_node_rels[localFaceNode];

          // extract nodal fields
          const double * coordL = stk::mesh::field_data(*coordinates_, nodeL );
          const double * coordR = stk::mesh::field_data(*coordinates_, nodeR );

          // extract point info for this ip - must matches the construction of the pInfo vector
          PointInfo *pInfo = faceInfoVec[ip];
          stk::mesh::Entity owningElement = pInfo->owningElement_;

          /*
          // get master element type for this contactInfo
          MasterElement *meSCS  = pInfo->meSCS_;
          const int nodesPerElement = meSCS->nodesPerElement_;
          std::vector <double > elemNodalVelocity(nodesPerElement*nDim_);
          std::vector <double > elemNodalCoords(nodesPerElement*nDim_);
          //std::vector <double > elemNodalGradP(nodesPerElement*nDim_);
          std::vector <double > elemNodalGradU(nodesPerElement*nDim_*nDim_);
          std::vector <double > shpfc(nodesPerElement);

          // gather element data
          stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(owningElement);
          const int num_elem_nodes = bulkData_->num_nodes(owningElement);
          for ( int ni = 0; ni < num_elem_nodes; ++ni ) {
            stk::mesh::Entity node = elem_node_rels[ni];
            // gather velocity and pressure gradient (conforms to interpolatePoint)
            const double *uNp1 = stk::mesh::field_data(velocityNp1, node );
            //const double *gradpNp1 = stk::mesh::field_data(dpdxNp1, node );
            const double *graduNp1 = stk::mesh::field_data(*dudx_, node );
            for (int j = 0; j < nDim_*nDim_; ++j) {
              std::cout << graduNp1[j] << "  ";
            }
            std::cout << std::endl;
            const double *coords = stk::mesh::field_data(*coordinates_, node );
            for ( int j = 0; j < nDim_; ++j ) {
              elemNodalVelocity[j*nodesPerElement+ni] = uNp1[j];
              elemNodalCoords[j*nodesPerElement+ni] = coords[j];
              //elemNodalGradP[j*nodesPerElement+ni] = gradpNp1[j];
            }
            // gather dudx tensor
            for (int i = 0; i < nDim_; ++i) {
              for (int j = 0; j < nDim_; ++j) {
                elemNodalGradU[ni*nDim_*nDim_+i*nDim_+j] = graduNp1[i*nDim_+j];
              }
            }
            //for ( int j = 0; j < nDim_*nDim_; ++j ) {
            //  elemNodalGradU[j*nodesPerElement+ni] = graduNp1[j];
            //}
          }
          
          // interpolate to elemental point location
          meSCS->interpolatePoint(
            nDim_,
            &(pInfo->isoParCoords_[0]),
            &elemNodalVelocity[0],
            &uProjected[0]);

            //meSCS->interpolatePoint(
            //nDim_,
            //&(pInfo->isoParCoords_[0]),
            //&elemNodalGradP[0],
            //&dpdxProjected[0]);

          meSCS->interpolatePoint(
            nDim_*nDim_,
            &(pInfo->isoParCoords_[0]),
            &elemNodalGradU[0],
            &dudxProjected[0]);


          // sanity check for coords
          meSCS->interpolatePoint(
            nDim_,
            &(pInfo->isoParCoords_[0]),
            &elemNodalCoords[0],
            &cProjected[0]);

          if ( provideOutput_ ) {
            for (int j = 0; j < nDim_; ++j ) 
              NaluEnv::self().naluOutput() << "Coords sanity check: " << cProjected[j] << " " << pInfo->pointCoordinates_[j] << std::endl;
          }
          */

          // zero out vector quantities; squeeze in aMag
          double aMag = 0.0;
          for ( int j = 0; j < nDim_; ++j ) {
            p_uBip[j] = 0.0;
            p_uBcBip[j] = 0.0;
            const double axj = areaVec[ipNdim+j];
            aMag += axj*axj;
          }
          aMag = std::sqrt(aMag);
          for ( int j = 0; j < nDim_*nDim_; ++j) {
            p_dudxBip[j] = 0.0;
          }

          // interpolate to bip
          double rhoBip = 0.0;
          double muBip = 0.0;
          const int ipNpf = ip*nodesPerFace;
          for ( int ic = 0; ic < nodesPerFace; ++ic ) {
            const double r = p_face_shape_function[ipNpf+ic];
            rhoBip += r*p_density[ic];
            muBip += r*p_viscosity[ic];
            const int icNdim = ic*nDim_;
            for ( int j = 0; j < nDim_; ++j ) {
              p_uBip[j] += r*p_velocityNp1[icNdim+j];
              p_uBcBip[j] += r*p_bcVelocity[icNdim+j];
            }
            const int offSetFNT = ic*nDim_*nDim_;
            for ( int j = 0; j<nDim_*nDim_; ++j ) {
              p_dudxBip[j] += r*p_dudx[offSetFNT+j];
            }
          }
          // form unit normal and determine yp (approximated by 1/4 distance along edge)
          double ypBip = 0.0;
          for ( int j = 0; j < nDim_; ++j ) {
            const double nj = areaVec[ipNdim+j]/aMag;
            const double ej = 0.25*(coordR[j] - coordL[j]);
            ypBip += nj*ej*nj*ej;
            p_unitNormal[j] = nj;
          }
          ypBip = std::sqrt(ypBip);
          wallNormalDistanceBip[ip] = ypBip;
          // form unit normal and determine yp
          //for ( int j = 0; j < nDim_; ++j ) {
          //  p_unitNormal[j] = areaVec[ipNdim+j]/aMag;
          // }
          
          //double ypBip = 0.03125; //pDistance;
          //wallNormalDistanceBip[ip] = ypBip;
          
          // assemble to nodal quantities
          double * assembledWallArea = stk::mesh::field_data(*assembledWallArea_, nearestNode );
          double * assembledWallNormalDistance = stk::mesh::field_data(*assembledWallNormalDistance_, nearestNode );
          
          *assembledWallArea += aMag;
          *assembledWallNormalDistance += aMag*ypBip;

          // determine tangential velocity
          std::vector<double> uTanNormal(nDim_);
          double uTangential = 0.0;
          for ( int i = 0; i < nDim_; ++i ) {
            double uiTan = 0.0;
            double uiBcTan = 0.0;
            for ( int j = 0; j < nDim_; ++j ) {
              const double ninj = p_unitNormal[i]*p_unitNormal[j];
              if ( i==j ) {
                const double om_nini = 1.0 - ninj;
                uiTan += om_nini*p_uBip[j];
                uiBcTan += om_nini*p_uBcBip[j];
              }
              else {
                uiTan -= ninj*p_uBip[j];
                uiBcTan -= ninj*p_uBcBip[j];
              }
            }
            uTangential += (uiTan-uiBcTan)*(uiTan-uiBcTan);
            uTanNormal[i] = uiTan;
          }
          uTangential = std::sqrt(uTangential);
          
          // provide an initial guess based on yplusCrit_ (more robust than a pure guess on utau)
          double utauGuess = yplusCrit_*muBip/rhoBip/ypBip;
          
          compute_utau(uTangential, ypBip, rhoBip, muBip, utauGuess);
          wallFrictionVelocityBip[ip] = utauGuess;
          tauLogBip[ip] = rhoBip*utauGuess*utauGuess;

          // partition tauWall_i based on unit normal for uTan
          double uTanMag = 0.0;
          for ( int i = 0; i < nDim_; ++i )
            uTanMag += uTanNormal[i]*uTanNormal[i];
          uTanMag = std::sqrt(uTanMag);
        
          // scatter it
          for ( int i = 0; i < nDim_; ++i )
            vectorTauWallBip[ipNdim+i] = rhoBip*utauGuess*utauGuess*uTanNormal[i]/uTanMag;


          //std::cout << "utau = " << utauGuess << std::endl;
          //std::cout << "uTangential = " << uTangential << std::endl;
          //std::cout << "ypBip = " << ypBip << std::endl;
          //std::cout << "muBip = " << muBip << std::endl;

          //std::cout << "BIP Velocity: " << std::endl;
          //for (int i = 0; i < nDim_; ++i) {
          //  std::cout << p_uBip[i] << " ";
          // }
          //std::cout << std::endl;
          //std::cout << "BIP dudx: " << std::endl;
          //for (int i = 0; i < nDim_*nDim_; ++i) {
          //  std::cout << p_dudxBip[i] << " ";
          // }
          //std::cout << std::endl;
          // Non-dimensionalize input features
          /*double dpdx_mag = 0.0;
          for ( int i = 0; i < nDim_; ++i ) {
            dpdx_mag += p_dpdxProjected[i] * p_dpdxProjected[i];
          }
          dpdx_mag = std::sqrt(dpdx_mag);*/
          double u_p = 0; // std::pow(muBip / (rhoBip * rhoBip) * dpdx_mag, 1.0/3.0);        
          const double u_tau_p = std::sqrt(utauGuess*utauGuess+u_p*u_p);
          for ( int i = 0; i < nDim_; ++i ) {
            p_uBip[i] /= std::max(1.0e-15, u_tau_p);
            //p_dpdxProjected[i] /= std::max(1.0e-15, rhoBip*rhoBip*u_tau_p*u_tau_p*u_tau_p/muBip);
          }
          for ( int i = 0; i < nDim_*nDim_; ++i ) {
            p_dudxBip[i] *= (muBip / std::max(1.0e-15,rhoBip*u_tau_p*u_tau_p));
          }
          //std::cout << "NONDIMENSIONAL BIP Velocity: " << std::endl;
          //for (int i = 0; i < nDim_; ++i) {
          //  std::cout << p_uBip[i] << " ";
          //}
          //std::cout << std::endl;
          //std::cout << "NONDIMENSIONAL BIP dudx: " << std::endl;
          //for (int i = 0; i < nDim_*nDim_; ++i) {
          //  std::cout << p_dudxBip[i] << " ";
          //}
          //std::cout << std::endl;

          // Form feature vectors
          std::vector<double> Pi(numPi_*nDim_,0.0);
          std::vector<double> lambda(numLambda_+1,0.0);
          double *p_Pi = &Pi[0];
          double *p_lambda = &lambda[0];
          double neg_unitNormal[3];
          for (int i=0; i<nDim_; ++i) neg_unitNormal[i] = - p_unitNormal[i];
          eval_features_T1(nDim_, &neg_unitNormal[0], p_uBip, p_dudxBip, p_lambda, p_Pi);

          //lambda = {0.000129585,  0.00108435,  -0.00108224,  1.3776e-06,  -4.55913e-07,  173.302,  -0.0494189,  -0.00707192,  0.00112446,  0.0090405,  -0.00626224,  0.000534281,  2.90137e-06,  -1.52769e-06,  1.43356e-07,  0.000532165,  0.303918,  -0.497627,  -0.00363531,  -0.303423};
          //Pi = {13.1392,  0.00232219,  -0,  0.0230583,  -0.00115369,  0.023021,  -0.815059,  -0.0011611,  0.0090405,  -0.00626224,  1,  0.00112446,  -0,  0,  -0,  -0,  -0.815059,  0.046042,  -0,  -0.00115369,  -0.0230583,  -0.0011611,  -13.1392,  -0.023021};
          lambda[numLambda_] = biasNodeValue_;
          //std::cout << "lambda : " << std::endl;
          //for (int i = 0; i < numLambda_; ++i) {
          //  std::cout << lambda[i] << "  ";
          // }
          //std::cout << std::endl;
          //std::cout << "Pi : " << std::endl;
          //for (int i = 0; i < numPi_*3; ++i) {
          //  std::cout << Pi[i] << "  ";
          //}
          //std::cout << std::endl;
          // Scale the scalar features
          std::vector<double> scaled_lambda(numLambda_);
          std::vector<double> scaled_Pi(numPi_*3);

          for (int i=0; i<numLambda_; ++i) {
            if (std::abs(fmin_[i] - fmax_[i]) > 1.0e-15) {
              scaled_lambda[i] = 2*(lambda[i] - fmin_[i]) / (fmax_[i] - fmin_[i]) - 1.0;
            } else {
              scaled_lambda[i] = -1;
            }
          }
          scaled_lambda[numLambda_] = biasNodeValue_;

          // Scale and reorder the tensor features
          int idxPi[24];
          for ( int i=0; i < numPi_; ++i ) {
            idxPi[3*i] = i;
            idxPi[3*i+1] = i + numPi_;
            idxPi[3*i+2] = i + 2*numPi_;
          }
          for (int i=0; i<numPi_; ++i) {
            scaled_Pi[idxPi[3*i]] = Pi[idxPi[3*i]] / sf_Pi_[i];
            scaled_Pi[idxPi[3*i+1]] = Pi[idxPi[3*i+1]] / sf_Pi_[i];
            scaled_Pi[idxPi[3*i+2]] = Pi[idxPi[3*i+2]] / sf_Pi_[i];
          }

          // Evaluation of the neural network 
          std::vector<double> matVec(numLayRow_[0],0.0);
          std::vector<double> ensembleMean(numLayRow_[numLayers_-1],0.0);
          for ( int i=0; i<numNetworks_; ++i ) {

            // Evaluate the first layer
            matVec.resize(numLayRow_[0]);
            matVec.assign(numLayRow_[0],0.0);
            std::vector<double> wt_ptr = weights_[i][0];
            for ( int j=0; j<numLayRow_[0]; ++j ) {
              int rowOff = j*numLayCol_[0];
              for ( int k=0; k<numLayCol_[0]; ++k ) {
                matVec[j] += wt_ptr[rowOff+k] * scaled_lambda[k];
              }
            }
            std::vector<double> layerResult(numLayRow_[0]+1); // +1 to make room for the bias value
            // ReLU
            for ( int j=0; j<numLayRow_[0]; ++j )
              layerResult[j] = ( matVec[j] > 0.0 ? matVec[j] : 0.0 );
            layerResult[numLayRow_[0]] = biasNodeValue_;

        // Loop over remaining layers
            for ( int l=1; l<numLayers_; ++l ) {
              matVec.resize(numLayRow_[l]);
              matVec.assign(numLayRow_[l],0.0);
              wt_ptr = weights_[i][l];
              for ( int j=0; j<numLayRow_[l]; ++j ) {
                int rowOff = j*numLayCol_[l];
                for ( int k=0; k<numLayCol_[l]; ++k ) {
                  matVec[j] += wt_ptr[rowOff+k] * layerResult[k];
                }
              }
              layerResult.resize(numLayRow_[l]+1);
           
              // If the last layer, no ReLU
              if (l == numLayers_-1) {
                for ( int j=0; j<numLayRow_[l]; ++j )
                  layerResult[j] = matVec[j];
              }
              else {
                // ReLU
                for ( int j=0; j<numLayRow_[l]; ++j )
                  layerResult[j] = ( matVec[j] > 0.0 ? matVec[j] : 0.0 );
                layerResult[numLayRow_[l]] = biasNodeValue_;
              }
            }
            
            for ( int j = 0; j<numLayRow_[numLayers_-1]; ++j )
              ensembleMean[j] += layerResult[j];

          } // end network loop

          for ( int j = 0; j < numLayRow_[numLayers_-1]; ++j )
            ensembleMean[j] /= numNetworks_;

          // Evaluate the neural network prediction for scaled tauw
          double tauw_ML_scaled[3];
          for ( int j = 0; j < 3; ++j ) {
            tauw_ML_scaled[j] = 0.0;
            for ( int k = 0; k < numPi_; ++k ) {
              int rowOffset = j*numPi_;
              tauw_ML_scaled[j] += ensembleMean[k] * scaled_Pi[rowOffset+k];
            }
          }

          // Unscale tauw and populate IP tauwall vector, friction velocity          
          double tauw_ML[3];
          for (int i=0; i<3; ++i) {
            tauw_ML[i] = tauw_ML_scaled[i] * sf_tau_;
            //std::cout << tauw_ML[i] << " ";
          }
          //std::cout << std::endl;
          
          /*vectorTauWallBip[ipNdim] = tauw_ML[0];
          vectorTauWallBip[ipNdim+1] = tauw_ML[1];
          vectorTauWallBip[ipNdim+2] = tauw_ML[2];
          wallFrictionVelocityBip[ip] = std::sqrt(std::sqrt((tauw_ML[0]*tauw_ML[0]+tauw_ML[1]*tauw_ML[1]+tauw_ML[2]*tauw_ML[2]))/rhoBip);*/
          
        }
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- set_data --------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::set_data( 
  double theDouble)
{
  projectedDistanceVec_.push_back(theDouble);
}

//--------------------------------------------------------------------------
//-------- compute_utau ----------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::compute_utau(
    const double &up, const double &yp,
    const double &density, const double &viscosity,
    double &utau )
{
  bool converged = false;

  const double A = elog_*density*yp/viscosity;

  for ( int k = 0; k < maxIteration_; ++k ) {

    const double wrk = std::log(A*utau);

    // evaluate F'

    const double fPrime = -(1.0+wrk);

    // evaluate function
    const double f = kappa_*up- utau*wrk;

    // update variable
    const double df = f/fPrime;

    utau -= df;
    if ( std::abs(df) < tolerance_ ) {
      converged = true;
      break;
    }
  }

  // report trouble
  if (!converged ) {
    NaluEnv::self().naluOutputP0() << "Issue with utau; not converged " << std::endl;
    NaluEnv::self().naluOutputP0() << up << " " << yp << " " << utau << std::endl;
  }
}

//--------------------------------------------------------------------------
//-------- eval_features----------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::eval_features_T1(int nDim, double *n, double *u, double *dudx, double *lambda, double *Pi)
{
  // Calculate the strain rate and rotation rate tensors
  std::vector<double> Sij(nDim*nDim);
  std::vector<double> Oij(nDim*nDim);
  for ( int i = 0; i < nDim; ++i ) {
    int offSet = i*nDim;
    for ( int j = 0; j < nDim; ++j ) {
      Sij[offSet+j] = 0.5 * ( dudx[offSet+j] + dudx[nDim*j+i] );
      Oij[offSet+j] = 0.5 * ( dudx[offSet+j] - dudx[nDim*j+i] );
    }
  }

  // Calculate some intermediate vectors
  std::vector<double> S2(nDim*nDim);
  square_matmat_mult(nDim, &Sij[0], &Sij[0], &S2[0]); // S^2
  std::vector<double> O2(nDim*nDim); 
  square_matmat_mult(nDim, &Oij[0], &Oij[0], &O2[0]); // O^2
  std::vector<double> S3(nDim*nDim);
  square_matmat_mult(nDim, &S2[0], &Sij[0], &S3[0]); // S^3
  std::vector<double> SO2(nDim*nDim);
  square_matmat_mult(nDim, &Sij[0], &O2[0], &SO2[0]); // S*O^2
  std::vector<double> SU(nDim);
  matvec(nDim, nDim, &Sij[0], u, &SU[0]); // S*U
  std::vector<double> permO(nDim);
  permutation_tensor(&Oij[0], &permO[0]); // Epsilon(O)
  std::vector<double> Sn(nDim);
  matvec(nDim, nDim, &Sij[0], n, &Sn[0]); // S*n
  std::vector<double> On(nDim);
  matvec(nDim, nDim, &Oij[0], n, &On[0]); // O*n
  std::vector<double> S2n(nDim); 
  matvec(nDim, nDim, &S2[0], n, &S2n[0]); // S^2*n
  std::vector<double> SO(nDim*nDim);
  square_matmat_mult(nDim, &Sij[0], &Oij[0], &SO[0]); // S*O
  std::vector<double> permSO(nDim);
  permutation_tensor(&SO[0], &permSO[0]); // Epsilon(S*O)
  double Sn_cross_S2n[3];
  cross_product(&Sn[0], &S2n[0], Sn_cross_S2n); // S*n x S^2*n
  std::vector<double> permSO2(nDim);
  permutation_tensor(&SO2[0], &permSO2[0]); // Epsilon(S*O^2)
  std::vector<double> SOn(nDim);
  matvec(nDim, nDim, &SO[0], n, &SOn[0]); // S*O*n
  double U_cross_SU[3];
  cross_product(u, &SU[0], U_cross_SU); // U x S*U
  double U_cross_Sn[3];
  cross_product(u, &Sn[0], U_cross_Sn); // U x S*n
  std::vector<double> OU(nDim);
  matvec(nDim, nDim, &Oij[0], u, &OU[0]); // O*U


  // Populate the lambdas
  lambda[0] = trace(nDim, &Sij[0]);        // lambda_1 = {S}
  lambda[1] = trace(nDim, &S2[0]);         // lambda_2 = {S^2}
  lambda[2] = trace(nDim, &O2[0]);         // lambda_3 = {O^2}
  lambda[3] = trace(nDim, &S3[0]);         // lambda_4 = {S^3}
  lambda[4] = trace(nDim, &SO2[0]);        // lambda_5 = {S*O^2}
  lambda[5] = dot(nDim, u, u);             // lambda_6 = U dot U
  lambda[6] = dot(nDim, u, &SU[0]);        // lambda_7 = U dot S*U
  lambda[7] = dot(nDim, u, &permO[0]);     // lambda_8 = U dot Epsilon(O)
  lambda[8] = dot(nDim, n, &Sn[0]);        // lambda_9 = n dot S*n
  lambda[9] = dot(nDim, n, u);             // lambda_10 = n dot u
  lambda[10] = dot(nDim, n, &permO[0]);    // lambda_11 = n dot Epsilon(O)
  lambda[11] = dot(nDim, n, &S2n[0]);      // lambda_12 = n dot S^2*n
  lambda[12] = dot(nDim, n, &permSO[0]);   // lambda_13 = n dot Epsilon(S*O)
  lambda[13] = dot(nDim, n, Sn_cross_S2n); // lambda_14 = n dot (S*n x S^2*n)
  lambda[14] = dot(nDim, n, &permSO2[0]);  // lambda_15 = n dot Epsilon(S*O^2)
  lambda[15] = dot(nDim, n, &SOn[0]);      // lambda_16 = n dot S*O*n
  lambda[16] = dot(nDim, n, &SU[0]);       // lambda_17 = n dot S*U
  lambda[17] = dot(nDim, n, U_cross_SU);   // lambda_18 = n dot (U x S*U)
  lambda[18] = dot(nDim, n, U_cross_Sn);   // lambda_19 = n dot (U x S*n)
  lambda[19] = dot(nDim, n, &OU[0]);       // lambda_20 = n dot O*U

  // Populate the Pi functions
  double crossprod[3];
  for ( int j=0; j < nDim; ++j ) {
    int offSet = j*numPi_;
    Pi[offSet] = u[j];           // Pi_1 = U
    Pi[offSet+1] = permO[j];    // Pi_2 = epsilon*O
    Pi[offSet+2] = n[j];         // Pi_3 = n
    Pi[offSet+3] = Sn[j];        // Pi_4 = S*n
    Pi[j*numPi_+5] = On[j];      // Pi_6 = Omega*n
  }

  cross_product(n, &Sn[0], crossprod);
  for ( int j=0; j < nDim; ++j ) {
    Pi[j*numPi_+4] = crossprod[j];   // Pi_5 = n x S*n
  }
  cross_product(n, u, crossprod);
  for ( int j=0; j < nDim; ++j ) {
    Pi[j*numPi_+6] = crossprod[j];    // Pi_7 = n x U
  } 
  cross_product(n, &On[0], crossprod);
  for ( int j=0; j < nDim; ++j ) {
    Pi[j*numPi_+7] = crossprod[j];    // Pi_8 = n x O*n
  } 

}

//--------------------------------------------------------------------------
//-------- cross_product ---------------------------------------------------
//--------------------------------------------------------------------------
// This is coded for nDim=3 only.
void
ComputeMLTauWallProjectedAlgorithm::cross_product(
  double *u, double *v, double *cross)
{
  cross[0] =   u[1]*v[2] - u[2]*v[1];
  cross[1] = -(u[0]*v[2] - u[2]*v[0]);
  cross[2] =   u[0]*v[1] - u[1]*v[0];
}

//--------------------------------------------------------------------------
//-------- permutation_tensor-----------------------------------------------
//--------------------------------------------------------------------------
// This is coded for nDim=3 only.
void
ComputeMLTauWallProjectedAlgorithm::permutation_tensor(
 double *T, double *p)
{
  p[0] = T[5] - T[7];
  p[1] = T[6] - T[2];
  p[2] = T[1] - T[3];
}

//--------------------------------------------------------------------------
//-------- matvec ----------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::matvec(
 int nr, int nc, double *M, double *v, double *out) 
{
  for ( int i = 0; i < nr; ++i ) {
    out[i] = 0.0;
    int offSet = i*nc;
    for ( int j = 0; j < nr; ++j ) {
      out[i] += M[offSet+j] * v[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- matmat_mult -----------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::square_matmat_mult(
 int nr, double *M, double *N, double *out)
{
  for ( int k = 0; k < nr*nr; ++k )
    out[k] = 0.0;

  for ( int i = 0; i < nr; ++i ) {
    int offSetRow = i*nr;
    for ( int j = 0; j < nr; ++j ) {
      int offSetCol = j;
      for ( int k = 0; k < nr; ++k ) {
	out[offSetRow+j] += M[offSetRow+k] * N[offSetCol+nr*k];
      }
    }
  }
}

double 
ComputeMLTauWallProjectedAlgorithm::trace(
 int n, double *M)
{
  double out = 0.0;
  for ( int i = 0; i < n; ++i ) {
    out += M[i*n+i];
  }
  return out;
}

double
ComputeMLTauWallProjectedAlgorithm::dot(
 int n, double *u, double *v)
{
  double out = 0.0;
  for ( int i = 0; i < n; ++i ) {
    out += u[i] * v[i];
  }
  return out;
}

void
ComputeMLTauWallProjectedAlgorithm::outer_product(
 int n, double *u, double *v, double *out)
{
  for ( int i = 0; i < n; ++i ) {
    int offset = i*n;
    for ( int j = 0; j < n; ++j ) {
      out[offset+j] = u[i]*v[j];
    }
  }
}

//--------------------------------------------------------------------------
//-------- initialize ------------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::initialize()
{
  
  // only process if first time or mesh motion is active
  if ( !firstInitialization_  && !realm_.has_mesh_motion() )
    return;
  
  // clear some of the search info
  boundingPointVec_.clear();
  boundingBoxVec_.clear();
  searchKeyPair_.clear();

  // initialize all ghosting data structures
  initialize_ghosting();
  
  // construct if the size is zero; reset always
  if ( pointInfoVec_.size() == 0 )
    construct_bounding_points();
  reset_point_info();
  
  // construct the bounding boxes
  construct_bounding_boxes();

  // coarse search (fills elemsToGhost_)
  coarse_search();

  manage_ghosting();

  // complete search
  complete_search();
  
  // set flag for the next possible time we are through the initialization method
  firstInitialization_ = false;
}

//--------------------------------------------------------------------------
//-------- initialize_ghosting ---------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::initialize_ghosting()
{
  // initialize need to ghost and elems to ghost
  needToGhostCount_ = 0;
  elemsToGhost_.clear();
  
  bulkData_->modification_begin();  
  if ( nullptr == wallFunctionGhosting_) {
    // create new ghosting
    std::string theGhostName = "nalu_wall_function_ghosting";
    wallFunctionGhosting_ = &(bulkData_->create_ghosting( theGhostName ));
  }
  else {
    bulkData_->destroy_ghosting(*wallFunctionGhosting_);
  }
  bulkData_->modification_end();
}

//--------------------------------------------------------------------------
//-------- construct_bounding_points --------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::construct_bounding_points()
{
  // hold the point location projected off the face integration points
  Point ipCoordinates;
  Point pointCoordinates;

  // field extraction
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // nodal fields to gather
  std::vector<double> ws_coordinates;
  std::vector<double> ws_face_shape_function;

  // fixed size
  std::vector<double> ws_unitNormal(nDim_,0.0);
  
  // need to keep track of some sort of local id for each gauss point...
  uint64_t localPointId = 0;
  
  // iterate over parts to allow for projected distance to vary per surface and defines ordering everywhere else
  for ( size_t pv = 0; pv < partVec_.size(); ++pv ) {
    
    // extract projected distance
    const double pDistance = projectedDistanceVec_[pv];
    
    // define selector (per part)
    stk::mesh::Selector s_locally_owned 
      = metaData_->locally_owned_part() &stk::mesh::Selector(*partVec_[pv]);
    
    stk::mesh::BucketVector const& face_buckets =
      realm_.get_buckets( metaData_->side_rank(), s_locally_owned );
    
    for ( stk::mesh::BucketVector::const_iterator ib = face_buckets.begin();
          ib != face_buckets.end() ; ++ib ) {
      stk::mesh::Bucket & b = **ib ;
      
      // face master element
      MasterElement *meFC = sierra::nalu::MasterElementRepo::get_surface_master_element(b.topology());
      const int nodesPerFace = b.topology().num_nodes();
      const int numScsBip = meFC->numIntPoints_;
      
      // algorithm related; element
      ws_coordinates.resize(nodesPerFace*nDim_);
      ws_face_shape_function.resize(numScsBip*nodesPerFace);
      
      // pointers
      double *p_coordinates = &ws_coordinates[0];
      double *p_face_shape_function = &ws_face_shape_function[0];
      
      // shape functions
      if ( useShifted_ )
        meFC->shifted_shape_fcn(&p_face_shape_function[0]);
      else
        meFC->shape_fcn(&p_face_shape_function[0]);
      
      const stk::mesh::Bucket::size_type length   = b.size();
      
      for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
        
        // get face
        stk::mesh::Entity face = b[k];
        
        // pointer to face data
        const double * areaVec = stk::mesh::field_data(*exposedAreaVec_, face);
        
        //======================================
        // gather nodal data off of face
        //======================================
        stk::mesh::Entity const * face_node_rels = bulkData_->begin_nodes(face);
        int num_face_nodes = bulkData_->num_nodes(face);
        // sanity check on num nodes
        ThrowAssert( num_face_nodes == nodesPerFace );
        for ( int ni = 0; ni < num_face_nodes; ++ni ) {
          stk::mesh::Entity node = face_node_rels[ni]; 
          // gather vectors
          double * coords = stk::mesh::field_data(*coordinates, node);
          const int niNdim = ni*nDim_;
          for ( int j=0; j < nDim_; ++j ) {
            p_coordinates[niNdim+j] = coords[j];
          }
        }
                        
        // set size for vector of points on this face
        std::vector<PointInfo *> faceInfoVec(numScsBip);
        for ( int ip = 0; ip < numScsBip; ++ip ) { 
          
          // compute area magnitude
          double aMag = 0.0;
          for ( int j = 0; j < nDim_; ++j ) {
            const double axj = areaVec[ip*nDim_+j];
            aMag += axj*axj;
          }
          aMag = std::sqrt(aMag);
          
          // compute normal (outward facing)
          for ( int j = 0; j < nDim_; ++j ) {
            ws_unitNormal[j] = areaVec[ip*nDim_+j]/aMag;
            ipCoordinates[j] = 0.0;
          }
          
          // interpolate coodinates to gauss point
          const int ipNpf = ip*nodesPerFace;
          for ( int ic = 0; ic < nodesPerFace; ++ic ) {
            const double r = p_face_shape_function[ipNpf+ic];
            for ( int j = 0; j < nDim_; ++j ) {
              ipCoordinates[j] += r*p_coordinates[ic*nDim_+j];
            }
          }
          
          // project in space (unit normal is outward facing, hence the -)
          for ( int j = 0; j < nDim_; ++j ) {
            pointCoordinates[j] = ipCoordinates[j] - pDistance*ws_unitNormal[j];
          }
          
          // setup ident for this point; use local integration point id
          uint64IdentProc theIdent(localPointId, NaluEnv::self().parallel_rank());
          
          // create the bounding point and push back
          boundingPoint bPoint(Point(pointCoordinates), theIdent);
          boundingPointVec_.push_back(bPoint);
          
          PointInfo *pInfo = new PointInfo(bPoint, localPointId, ipCoordinates, pointCoordinates, nDim_);
          faceInfoVec[ip] = pInfo;
          localPointId++;
        }
        
        // push them all back
        pointInfoVec_.push_back(faceInfoVec);
      }
    }
  }
}

//--------------------------------------------------------------------------
//-------- construct_bounding_boxes ----------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::construct_bounding_boxes()
{
  // extract coordinates
  VectorFieldType *coordinates
    = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
  
  // setup data structures for search
  Point minCorner, maxCorner;

  // deal with element part vector
  stk::mesh::PartVector elemBlockPartVec;
  const stk::mesh::PartVector allParts = metaData_->get_parts();
  for ( size_t k = 0; k < allParts.size(); ++k ) {
    if ( stk::mesh::is_element_block(*allParts[k]) ) {
      elemBlockPartVec.push_back(allParts[k]);
    }
  }
  
  // selector
  stk::mesh::Selector s_locally_owned_union
    = metaData_->locally_owned_part() &stk::mesh::selectUnion(elemBlockPartVec);
  stk::mesh::BucketVector const &elem_buckets 
    = bulkData_->get_buckets( stk::topology::ELEMENT_RANK, s_locally_owned_union );
  
  for ( stk::mesh::BucketVector::const_iterator ib = elem_buckets.begin();
        ib != elem_buckets.end() ; ++ib ) {
    stk::mesh::Bucket & b = **ib;
    
    const stk::mesh::Bucket::size_type length   = b.size();
    for ( stk::mesh::Bucket::size_type k = 0 ; k < length ; ++k ) {
      
      // get element
      stk::mesh::Entity element = b[k];
      
      // initialize max and min
      for (int j = 0; j < nDim_; ++j ) {
        minCorner[j] = +1.0e16;
        maxCorner[j] = -1.0e16;
      }
      
      // extract elem_node_relations
      stk::mesh::Entity const* elem_node_rels = bulkData_->begin_nodes(element);
      const int num_nodes = bulkData_->num_nodes(element);
      
      for ( int ni = 0; ni < num_nodes; ++ni ) {
        stk::mesh::Entity node = elem_node_rels[ni];
        
        // pointers to real data
        const double * coords = stk::mesh::field_data(*coordinates, node );
        
        // check max/min
        for ( int j = 0; j < nDim_; ++j ) {
          minCorner[j] = std::min(minCorner[j], coords[j]);
          maxCorner[j] = std::max(maxCorner[j], coords[j]);
        }
      }
      
      // setup ident
      uint64IdentProc theIdent(bulkData_->identifier(element), NaluEnv::self().parallel_rank());
      
      // expand the box 
      for ( int i = 0; i < nDim_; ++i ) {
        const double theMin = minCorner[i];
        const double theMax = maxCorner[i];
        const double increment = expandBoxPercentage_*(theMax - theMin);
        minCorner[i] -= increment;
        maxCorner[i] += increment;
      }

      // correct for 2d
      //if ( nDim_ == 2 ) {
      //  minCorner[2] = -1.0;
      //  maxCorner[2] = +1.0;
      // }
      
      // create the bounding box and push back
      boundingBox theBox(Box(minCorner,maxCorner), theIdent);
      boundingBoxVec_.push_back(theBox);
    }
  }
}

//--------------------------------------------------------------------------
//-------- reset_point_info -----------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::reset_point_info()
{
  std::vector<std::vector<PointInfo*> >::iterator ii;
  for( ii=pointInfoVec_.begin(); ii!=pointInfoVec_.end(); ++ii ) {
    std::vector<PointInfo *> &theVec = (*ii);    
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      PointInfo *pInfo = theVec[k];
      pInfo->bestX_ = pInfo->bestXRef_;
    }
  }
}

//--------------------------------------------------------------------------
//-------- coarse_search ---------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::coarse_search()
{
  // first coarse search
  stk::search::coarse_search(boundingPointVec_, boundingBoxVec_,
                             searchMethod_, NaluEnv::self().parallel_comm(), searchKeyPair_);
  
  // sort the product of the search
  std::sort(searchKeyPair_.begin(), searchKeyPair_.end(), lessThan());
  
  // now determine elements to ghost
  std::vector<std::pair<boundingPoint::second_type, boundingBox::second_type> >::const_iterator ii;  
  for ( ii=searchKeyPair_.begin(); ii!=searchKeyPair_.end(); ++ii ) {
    const uint64_t theBox = ii->second.id();
    unsigned theRank = NaluEnv::self().parallel_rank();
    const unsigned pt_proc = ii->first.proc();
    const unsigned box_proc = ii->second.proc();
    if ( (box_proc == theRank) && (pt_proc != theRank) ) {
      
      // find the element
      stk::mesh::Entity theElemMeshObj = bulkData_->get_entity(stk::topology::ELEMENT_RANK, theBox);
      if ( !(bulkData_->is_valid(theElemMeshObj)) )
        throw std::runtime_error("no valid entry for element");

      // new element to ghost counter
      needToGhostCount_++;
      
      // deal with elements to push back to be ghosted
      stk::mesh::EntityProc theElemPair(theElemMeshObj, pt_proc);
      elemsToGhost_.push_back(theElemPair); 
    }
  }
}

//--------------------------------------------------------------------------
//-------- manage_ghosting -------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::manage_ghosting()
{  
  // check for ghosting need
  size_t g_needToGhostCount = 0;
  stk::all_reduce_sum(NaluEnv::self().parallel_comm(), &needToGhostCount_, &g_needToGhostCount, 1);
  if (g_needToGhostCount > 0) {
    
    NaluEnv::self().naluOutputP0() << "Projected LOW alg will ghost a number of entities: "
                                   << g_needToGhostCount  << std::endl;
    
    bulkData_->modification_begin();
    bulkData_->change_ghosting( *wallFunctionGhosting_, elemsToGhost_);
    bulkData_->modification_end();
    // no linsys contribution and, hence, no populate_ghost_comm_procs()
  }
  else {
    NaluEnv::self().naluOutputP0() << "Projected LOW alg will NOT ghost entities: " << std::endl;
  }
  
  // ensure that the coordinates for the ghosted elements (required for the fine search) are up-to-date
  if (g_needToGhostCount > 0 ) {
    VectorFieldType *coordinates 
      = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());
    std::vector<const stk::mesh::FieldBase*> fieldVec = {coordinates};
    stk::mesh::communicate_field_data(*wallFunctionGhosting_, fieldVec);
  }
}

//--------------------------------------------------------------------------
//-------- complete_search--------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::complete_search()
{
  // fields
  VectorFieldType *coordinates = metaData_->get_field<VectorFieldType>(stk::topology::NODE_RANK, realm_.get_coordinates_name());

  // coordinates
  std::vector<double> isoParCoords(nDim_);
  std::vector<double> pointCoords(nDim_);

  // invert the process... Loop over InfoVec_ and query searchKeyPair_ for this information (avoids a map)
  std::vector<PointInfo *> problemInfoVec;
  std::vector<std::vector<PointInfo *> >::iterator ii;
  for( ii=pointInfoVec_.begin(); ii!=pointInfoVec_.end(); ++ii ) {
    std::vector<PointInfo *> &theVec = (*ii);
    for ( size_t k = 0; k < theVec.size(); ++k ) {
      
      PointInfo *pInfo = theVec[k];
      const uint64_t localPointId  = pInfo->localPointId_; 
      for ( int j = 0; j < nDim_; ++j )
        pointCoords[j] = pInfo->pointCoordinates_[j];
      
      std::pair <std::vector<std::pair<uint64IdentProc, uint64IdentProc> >::const_iterator, std::vector<std::pair<uint64IdentProc, uint64IdentProc> >::const_iterator > 
        p2 = std::equal_range(searchKeyPair_.begin(), searchKeyPair_.end(), localPointId, compareId());
      
      if ( p2.first == p2.second ) {
        problemInfoVec.push_back(pInfo);        
      }
      else {
        for (std::vector<std::pair<uint64IdentProc, uint64IdentProc> >::const_iterator jj = p2.first; jj != p2.second; ++jj ) {
          
          const uint64_t theBox = jj->second.id();
          const unsigned theRank = NaluEnv::self().parallel_rank();
          const unsigned pt_proc = jj->first.proc();
        
          // check if I own the point...
          if ( theRank == pt_proc ) {

            // proceed as required; all elements should have already been ghosted via the coarse search
            stk::mesh::Entity candidateElement = bulkData_->get_entity(stk::topology::ELEMENT_RANK, theBox);
            if ( !(bulkData_->is_valid(candidateElement)) )
              throw std::runtime_error("no valid entry for element");
            
            int elemIsGhosted = bulkData_->bucket(candidateElement).owned() ? 0 : 1;
                        
            // now load the elemental nodal coords
            stk::mesh::Entity const * elem_node_rels = bulkData_->begin_nodes(candidateElement);
            int num_nodes = bulkData_->num_nodes(candidateElement);            
            std::vector<double> elementCoords(nDim_*num_nodes);
            
            for ( int ni = 0; ni < num_nodes; ++ni ) {
              stk::mesh::Entity node = elem_node_rels[ni];
              // gather coordinates (conforms to isInElement)
              const double * coords =  stk::mesh::field_data(*coordinates, node);
              for ( int j = 0; j < nDim_; ++j ) {
                elementCoords[j*num_nodes+ni] = coords[j];
              }
            }
            
            // extract the topo from this element...
            const stk::topology elemTopo = bulkData_->bucket(candidateElement).topology();
            MasterElement *meSCS = sierra::nalu::MasterElementRepo::get_surface_master_element(elemTopo);

            const double nearestDistance = meSCS->isInElement(&elementCoords[0],
                                                              &(pointCoords[0]),
                                                              &(isoParCoords[0]));

            // check if this element is the best
            if ( nearestDistance < pInfo->bestX_ ) {
              pInfo->owningElement_ = candidateElement;
              pInfo->meSCS_ = meSCS;
              pInfo->isoParCoords_ = isoParCoords;
              pInfo->bestX_ = nearestDistance;
              pInfo->elemIsGhosted_ = elemIsGhosted;
            }
          }
          else {
            // not this proc's issue
          }
        }
      }
    }
  }
  
  if ( provideOutput_ ) {
    
    // provide output
    for( ii=pointInfoVec_.begin(); ii!=pointInfoVec_.end(); ++ii ) {
      std::vector<PointInfo *> &theVec = (*ii);
      for ( size_t k = 0; k < theVec.size(); ++k ) {        
        provide_output(theVec[k], false);        
      }
    }
    
    // sanity check on the elements provided in the bounding box
    for ( size_t k = 0; k < boundingBoxVec_.size(); ++k ) {   
      NaluEnv::self().naluOutput() << "element ids provided to search: " << boundingBoxVec_[k].second.id() << std::endl; 
      Box bB = boundingBoxVec_[k].first;
      NaluEnv::self().naluOutput() <<  ".... x: " << bB.get_x_min() << " " << bB.get_x_max() << std::endl;
      NaluEnv::self().naluOutput() <<  ".... y: " << bB.get_y_min() << " " << bB.get_y_max() << std::endl;
      NaluEnv::self().naluOutput() <<  ".... z: " << bB.get_z_min() << " " << bB.get_z_max() << std::endl;
    }
  }

  // check if there was a problem
  if ( problemInfoVec.size() > 0 ) {
    NaluEnv::self().naluOutputP0() << "there was BIG PROBLEM with problemInfoVec " << std::endl; 
    
    for ( size_t k = 0; k < problemInfoVec.size(); ++k ) {   
      provide_output(problemInfoVec[k], true);        
    }
    
    // sanity check on the elements provided in the bounding box
    for ( size_t k = 0; k < boundingBoxVec_.size(); ++k ) {   
      NaluEnv::self().naluOutput() << " BIG PROBLEM: element ids provided to search: " << boundingBoxVec_[k].second.id() << std::endl; 
      Box bB = boundingBoxVec_[k].first;
      NaluEnv::self().naluOutput() <<  ".... x: " << bB.get_x_min() << " " << bB.get_x_max() << std::endl;
      NaluEnv::self().naluOutput() <<  ".... y: " << bB.get_y_min() << " " << bB.get_y_max() << std::endl;
      NaluEnv::self().naluOutput() <<  ".... z: " << bB.get_z_min() << " " << bB.get_z_max() << std::endl;
    }
    throw std::runtime_error("ComputeMLTauWallProjectedAlgorithm::search_error()");
  }
}

//--------------------------------------------------------------------------
//-------- provide_output --------------------------------------------------
//--------------------------------------------------------------------------
void
ComputeMLTauWallProjectedAlgorithm::provide_output( 
  const PointInfo *pInfo,
  const bool problemPoint)
{
  const uint64_t localId = pInfo->localPointId_;
  stk::mesh::Entity theElem = pInfo->owningElement_;
  
  NaluEnv::self().naluOutput() << "...Review for Point ip: " << localId << std::endl;
  if ( problemPoint )
    NaluEnv::self().naluOutput() << "   BAD POINT" << std::endl;
  
  NaluEnv::self().naluOutput() << "   owning element id:    " << bulkData_->identifier(theElem) << std::endl;
  
  NaluEnv::self().naluOutput() << "   face coordinates: ";
  for ( int j = 0; j < nDim_; ++j )
    NaluEnv::self().naluOutput() << " " << pInfo->ipCoordinates_[j] << " ";
  NaluEnv::self().naluOutput() << std::endl;
  
  NaluEnv::self().naluOutput() << "   proj coordinates: ";
  for ( int j = 0; j < nDim_; ++j )
    NaluEnv::self().naluOutput() << " " << pInfo->pointCoordinates_[j] << " ";
  NaluEnv::self().naluOutput() << std::endl;
  
  NaluEnv::self().naluOutput() << "   point coordinates: " << std::endl;
  Point bP = pInfo->bPoint_.first;
  NaluEnv::self().naluOutput() <<  ".... x: " << bP.get_x_min() << " " << bP.get_x_max() << std::endl;
  NaluEnv::self().naluOutput() <<  ".... y: " << bP.get_y_min() << " " << bP.get_y_max() << std::endl;
  NaluEnv::self().naluOutput() <<  ".... z: " << bP.get_z_min() << " " << bP.get_z_max() << std::endl;   
}

} // namespace nalu
} // namespace Sierra
