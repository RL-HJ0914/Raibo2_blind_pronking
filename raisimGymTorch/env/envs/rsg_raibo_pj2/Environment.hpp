// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

// raisimGymTorch include
#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "RaiboController.hpp"
#include "RandomHeightMapGenerator.hpp"

namespace raisim {

class ENVIRONMENT {

 public:

  explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int id) :
      resourceDir_(resourceDir), visualizable_(visualizable) {
    setSeed(id);

    /// add objects
    raibo_ = world_.addArticulatedSystem(resourceDir + "/raibo2/urdf/raibo2.urdf");
    raibo_->setName("robot");
    raibo_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    jointLimit_ = raibo_->getJointLimits();
    maxForwardVel_ = 1.0;

    /// create controller
    controller_.create(&world_);
    nominalJointPgain_ = controller_.jointPgain_.tail(nJoints_)[0];
    nominalJointDgain_ = controller_.jointDgain_.tail(nJoints_)[0];

    /// indicies of the foot frame
    footFrameIndicies_[0] = raibo_->getFrameIdxByName("LF_S2F");
    footFrameIndicies_[1] = raibo_->getFrameIdxByName("RF_S2F");
    footFrameIndicies_[2] = raibo_->getFrameIdxByName("LH_S2F");
    footFrameIndicies_[3] = raibo_->getFrameIdxByName("RH_S2F");
    RSFATAL_IF(std::any_of(footFrameIndicies_.begin(), footFrameIndicies_.end(), [](int i){return i < 0;}), "footFrameIndicies_ not found")
    footMaterialName_ = {"LF_FOOT_MATERIAL", "RF_FOOT_MATERIAL", "LH_FOOT_MATERIAL", "RH_FOOT_MATERIAL"};

    /// set time step
    simulation_dt_ = controller_.getSimDt();
    control_dt_ = controller_.getConDt();

    /// read config
    READ_YAML(double, maxTime_, cfg["max_time"])
    READ_YAML(int, ip, cfg["ip"])
    READ_YAML(int, iterPerCurriculumUpdate, cfg["curriculum"]["iteration_per_update"])
    READ_YAML(int, iterPerTerrainCurriculumUpdate, cfg["curriculum"]["iteration_per_terrain_update"])
    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["initial_factor"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["decay_factor"])
    READ_YAML(double, terrainCurriculumFactor_, cfg["curriculum"]["terrain_initial_factor"])
    READ_YAML(double, terrainCurriculumDecayFactor_, cfg["curriculum"]["terrain_decay_factor"])
    READ_YAML(bool, testMode_, cfg["randomization"]["test_mode"])
    READ_YAML(bool, terrainRandomization_, cfg["randomization"]["terrain_randomization"])
    READ_YAML(bool, observationRandomization_, cfg["randomization"]["observation_randomization"])
    READ_YAML(bool, jointFrictionRandomization_, cfg["randomization"]["joint_friction_randomization"])
    READ_YAML(bool, voltageRandomization_, cfg["randomization"]["voltage_randomization"])
    READ_YAML(bool, simulationDtRandomization_, cfg["randomization"]["simulation_dt_randomization"])
    READ_YAML(bool, gainRandomization_, cfg["randomization"]["gain_randomization"])
    READ_YAML(bool, kinematicsRandomization_, cfg["randomization"]["kinematics_randomization"])
    READ_YAML(bool, baseInertiaParameterRandomization_, cfg["randomization"]["base_inertia_parameter_randomization"])

    if (baseInertiaParameterRandomization_) {
      auto nominalBase = raibo_->getLink("base");
      nominalMass_ = nominalBase.getWeight();
      nominalCoM_ = nominalBase.getComPositionInParentFrame();
      nominalI_ = nominalBase.getInertia();
      Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(nominalI_.e());
      nominalD_ = eigen_solver.eigenvalues().real();
      nominalRot_ = eigen_solver.eigenvectors().real();
    }


    /// create heightmap
    groundType_ = RandomHeightMapGenerator::GroundType::FLAT;
    groundTypeVector_ << 1.0, 0.0, 0.0;
    heightMap_ = terrainGenerator_.generateTerrain(&world_,
                                                   RandomHeightMapGenerator::GroundType::FLAT, gen_, uniDist_);
    groundType_ = RandomHeightMapGenerator::GroundType(id % terrainTypeNum_);
    if (groundType_ == RandomHeightMapGenerator::GroundType::TUK) maxForwardVel_ = 2.0;

    /// get robot data
    gcDim_ = int(raibo_->getGeneralizedCoordinateDim());
    gvDim_ = int(raibo_->getDOF());

    /// initialize containers
    gc_init_.setZero(gcDim_);
    gv_init_.setZero(gvDim_);
    nominalJointConfig_.setZero(nJoints_);
    gc_init_from_.setZero(gcDim_);
    gv_init_from_.setZero(gvDim_);
    clippedGenForce_.setZero(gvDim_);
    frictionTorque_.setZero(gvDim_);
    jointFrictions_.setZero(nJoints_);

    /// this is nominal configuration of raibo2
    nominalJointConfig_<< 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195;
    gc_init_.head(7) << 0, 0, 0.481, 1.0, 0.0, 0.0, 0.0;
    gc_init_.tail(12) = nominalJointConfig_;
    gc_init_from_ = gc_init_;
    raibo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    // Reward coefficients
    controller_.setRewardConfig(cfg);

    // visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(&world_);
      server_->focusOn(raibo_);
      if(ip==8080) server_->launchServer(8080);
      else server_->launchServer(98);

    }
  }

  ~ENVIRONMENT() { if (server_) server_->killServer(); }
  void init () { }
  void close () { }
  void setSimulationTimeStep(double dt) {
    world_.setTimeStep(dt);
    controller_.setSimDt(dt);
    simulation_dt_ = controller_.getSimDt();
  };
  void setControlTimeStep(double dt) {
    controller_.setConDt(dt);
    control_dt_ = controller_.getConDt();
  };
  void turnOffVisualization() { server_->hibernate(); }
  void turnOnVisualization() { server_->wakeup(); }
  void startRecordingVideo(const std::string& videoName ) { server_->startRecordingVideo(videoName); }
  void stopRecordingVideo() { server_->stopRecordingVideo(); }
  const std::vector<std::string>& getStepDataTag() { return controller_.getStepDataTag(); }
  const Eigen::VectorXd& getStepData() { return controller_.getStepData(); }

  void reset() {
    // terrain randomization
    if (terrainRandomization_  && (iter_ - 1) % iterPerTerrainCurriculumUpdate == 0) {
      terrainRandomization();
    }

    // voltage randomization
    if (voltageRandomization_) {
      voltage_ = 60.0 + uniDist_(gen_) * 20.0;
      jointVelLimit_ = 1400. / 60. * 2. * M_PI / 48. * voltage_ / 7.15 * 0.93; // 93% efficiency

      jointVelClipStart_ = jointVelLimit_ - jointVelClipRange_;
      raisim::VecDyn jointVelLimits(13);
      jointVelLimits.setZero();
      jointVelLimits.e().tail(nJoints_) = Eigen::VectorXd::Constant(12, jointVelLimit_);
      raibo_->setJointVelocityLimits(jointVelLimits);
    }

    // body position
    gc_init_.head(2).setZero();

    // orientation
    raisim::Mat<3,3> rotMat, yawRot, pitchRollMat;
    raisim::Vec<4> quaternion;
    raisim::Vec<3> axis = {normDist_(gen_), normDist_(gen_), normDist_(gen_)};
    axis /= axis.norm();
    raisim::angleAxisToRotMat(axis, normDist_(gen_) * 0.2, pitchRollMat);
    raisim::angleAxisToRotMat({0, 0, 1}, uniDist_(gen_) * 2. * M_PI, yawRot);
    if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
      auto normalVector = terrainGenerator_.normalVector_;
      Eigen::Vector3d zAxis{0, 0, 1};
      axis.e() = zAxis.cross(normalVector);
      if (terrainLevel_ == 0.0) axis.e() << 0, 0, 1;
      axis /= axis.norm();
      auto theta = std::acos(normalVector(2));
      skewSymMat(axis, pitchRollMat);
      pitchRollMat.e() = Eigen::Matrix3d::Identity() + std::sin(theta) * pitchRollMat.e() + (1 - std::cos(theta)) * pitchRollMat.e() * pitchRollMat.e();
//      raisim::angleAxisToRotMat({0,0,1}, uniDist_(gen_) * 2. * M_PI, yawRot);
      raisim::angleAxisToRotMat({0,0,1}, M_PI, yawRot);
    }
    rotMat = pitchRollMat * yawRot;
    raisim::rotMatToQuat(rotMat, quaternion);
    gc_init_.segment(3, 4) = quaternion.e();

    // joint angles
    for (int i = 0 ; i < nJoints_; i++) {
      gc_init_[i + 7] = nominalJointConfig_[i] + 0.3 * normDist_(gen_) * (1.5 - 0.5 * terrainLevel_);
      if (gc_init_[i + 7] < jointLimit_[i + 6][0] + 0.2) {
        gc_init_[i + 7] = jointLimit_[i + 6][0] + 0.2;
      }
      if (gc_init_[i + 7] > jointLimit_[i + 6][1] - 0.2) {
        gc_init_[i + 7] = jointLimit_[i + 6][1] - 0.2;
      }
    }

    /// randomize generalized velocities
    raisim::Vec<3> bodyVel_b, bodyVel_w, bodyAng_w;
    bodyVel_b.setZero();
    bodyVel_w.setZero();
    bodyAng_w.setZero();
    Eigen::VectorXd jointVel(12);
    jointVel.setZero();

    if (groundType_ != RandomHeightMapGenerator::GroundType::SLOPE) {
      bodyVel_b[0] = 0.6 * normDist_(gen_) * curriculumFactor_;
      bodyVel_b[1] = 0.6 * normDist_(gen_) * curriculumFactor_;
      bodyVel_b[2] = 0.3 * normDist_(gen_) * curriculumFactor_;
      bodyVel_b *= (2 - terrainLevel_);
      raisim::matvecmul(rotMat, bodyVel_b, bodyVel_w);
      for (int i = 0; i < 3; i++) bodyAng_w[i] = 0.4 * normDist_(gen_) * curriculumFactor_ * (2 - terrainLevel_);
      for (int i = 0; i < nJoints_; i++) jointVel[i] = 3. * normDist_(gen_) * curriculumFactor_ * (2 - terrainLevel_);
    }

    // combine
    gv_init_ << bodyVel_w.e(), bodyAng_w.e(), jointVel;

    // randomly initialize from previous trajectories
    if (uniDist_(gen_) < 0.25 && groundType_ != RandomHeightMapGenerator::GroundType::SLOPE) {
      gc_init_ = gc_init_from_;
      gv_init_ = gv_init_from_;
      gc_init_.head(2).setZero();
    }

    // command
    const bool standingMode = normDist_(gen_) > 1.3;
    controller_.setStandingMode(standingMode);
    double commandX = 0.;
    if (standingMode) {
      command_.setZero();
    } else {
      do {
        commandX = (((maxForwardVel_ - maxForwardVelAtMaxTerrainLevel_) * std::exp(-3. * terrainLevel_) + maxForwardVelAtMaxTerrainLevel_ + 1.0) * uniDist_(gen_) - 1.0);
        if (std::abs(commandX) < 0.3) commandX = 0.0;
        command_ << commandX * curriculumFactor_, /// ~ U(-1.0, maxForwardVel_)
            1.0 * 2. * (uniDist_(gen_) - 0.5) * curriculumFactor_ * (uniDist_(gen_) < 0.7 ? 0 : 1), /// ~ U(-1.0, 1.0)
            1.0 * 2. * (uniDist_(gen_) - 0.5) * curriculumFactor_ * (uniDist_(gen_) < 0.4 ? 0 : 1); /// ~ U(-1.0, 1.0)
      } while(command_.norm() < 0.3);
    }
    controller_.setCommand(command_);

    if (kinematicsRandomization_) {
      kinematicsRandomization();
    }

    if (testMode_) {
      if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
        gc_init_ << 0, 0, 0.481, quaternion.e(), nominalJointConfig_;
      }
      else {
        gc_init_ << 0, 0, 0.481, 1, 0, 0, 0, nominalJointConfig_;
      }
      gv_init_.setZero();
      world_.addGround(0.001);
    }

    raibo_->setGeneralizedCoordinate(gc_init_);

    // keep one foot on the terrain
    raisim::Vec<3> footPosition;
    double maxNecessaryShift = -1e20; /// some arbitrary high negative value
    for (auto & foot: footFrameIndicies_) {
      raibo_->getFramePosition(foot, footPosition);
      double terrainHeightMinusFootPosition = heightMap_->getHeight(footPosition[0], footPosition[1]) - footPosition[2];
      maxNecessaryShift = maxNecessaryShift > terrainHeightMinusFootPosition ? maxNecessaryShift : terrainHeightMinusFootPosition;
    }
    gc_init_[2] += maxNecessaryShift + 0.03;

//    if (gainRandomization_) {
//      gainRandomization();
//    }

    if (jointFrictionRandomization_) {
      jointFrictionRandomization();
    }

    if (baseInertiaParameterRandomization_) {
      baseInertiaParameterRandomization();
    }

    /// set the state
    raibo_->setState(gc_init_, gv_init_); /// set it again to ensure that foot is in contact
    controller_.reset(gen_, normDist_);
    controller_.updateStateVariables(heightMap_, true);

    for (auto & slip : slip_) slip = false;
  }

  inline void terrainRandomization() {
    if(server_) server_->lockVisualizationServerMutex();
    for (auto & obj : world_.getObjList()) {
      if (obj->getName() != "robot") {
        world_.removeObject(obj);
      }
    }
    if(server_) server_->unlockVisualizationServerMutex();

    terrainLevel_ = terrainCurriculumFactor_ * uniDist_(gen_);
    Eigen::Vector4d terrainParams;
    terrainParams(0) = 2 * maxTime_ * maxForwardVel_ - (2 * maxTime_ * maxForwardVel_ - 20) * terrainLevel_;
    if (groundType_ == RandomHeightMapGenerator::GroundType::HEIGHT_MAP) {
      terrainParams(1) = 0.2 + 0.4 * uniDist_(gen_); /// frequency ~ U(0.2, 1.0)
      terrainParams(2) = 0.2 + 0.4 * terrainLevel_; /// amplitude 0.2 -> 1.4
      groundTypeVector_ << 1.0, 0.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::HEIGHT_MAP_DISCRETE) {
      terrainParams(1) = 0.2 + 0.5 * terrainLevel_; /// amplitude 0.2 -> 1.2
      terrainParams(2) = 0.01 + 0.03 * uniDist_(gen_); /// step size ~ U(0.02, 0.15)
      groundTypeVector_ << 1.0, 0.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::STEPS) {
      terrainParams(1) = 0.1 + 0.4 * uniDist_(gen_); /// width ~ U(0.1, 0.5)
      terrainParams(2) = 0.01 + 0.03 * terrainLevel_; /// height 0.02 -> 0.18
      groundTypeVector_ << 0.0, 1.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::STEPS_INCLINE) {
      terrainParams(1) = 0.02 + 0.16 * terrainLevel_; /// roughness ~ 0.02 -> 0.18
      terrainParams(2) = 0.02 + 0.07 * terrainLevel_; /// height 0.02 -> 0.09
      groundTypeVector_ << 1.0, 0.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::STAIRS) {
      terrainParams(1) = 0.28 + 0.04 * uniDist_(gen_); /// width ~ U(0.28, 0.32)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_; /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// slope 35 deg
      groundTypeVector_ << 0.0, 1.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::NOSING_STAIRS) {
      terrainParams(1) = 0.28 + 0.04 * uniDist_(gen_); /// width ~ U(0.28, 0.32)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_; /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// slope 35 deg
      groundTypeVector_ << 0.0, 0.0, 1.0;
    }
    if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
      terrainParams(1) = 0.2 + 0.6 * uniDist_(gen_); /// frequency ~ U(0.2, 0.8)
      terrainParams(2) = 0.2 + 0.4 * terrainLevel_; /// amplitude 0.2 -> 0.6
      terrainParams(3) = 10.0 / 180 * M_PI * terrainLevel_; /// 35deg
      groundTypeVector_ << 1.0, 0.0, 0.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::STAIRS3) {
      terrainParams(1) = 0.25 + 0.1 * uniDist_(gen_); /// width ~ U(0.25, 0.35)
      terrainParams(2) = 0.02 + 0.16 * terrainLevel_; /// height 0.02 -> 0.18
      terrainParams(3) = 35.0 / 180 * M_PI * terrainLevel_; /// 35deg
      groundTypeVector_ << 0.0, 0.0, 1.0;
    }
    else if (groundType_ == RandomHeightMapGenerator::GroundType::TUK) {
      terrainParams(1) = 0.15 * terrainLevel_; /// height 0 -> 0.15
      groundTypeVector_ << 0.0, 0.0, 1.0;
    }
    terrainGenerator_.setTerrainParams(groundType_, terrainParams);
    if(server_) server_->lockVisualizationServerMutex();
    heightMap_ = terrainGenerator_.generateTerrain(&world_, groundType_, gen_, uniDist_);
    if(server_) server_->unlockVisualizationServerMutex();

    if (groundType_ == RandomHeightMapGenerator::GroundType::SLOPE) {
      double minFrictionCoeff = std::max(0.4, std::tan(terrainParams(3)) * 1.3);
      double interval = 2.0 - minFrictionCoeff;
      groundFrictionCoeff_ = minFrictionCoeff + interval * uniDist_(gen_); /// c_f ~ U(minFrictionCoeff, 2.0)
      world_.setDefaultMaterial(groundFrictionCoeff_, 0.0, 0.01);
      for (int i = 0; i < 4; i++) {
        world_.setMaterialPairProp(footMaterialName_[i], heightMap_->getCollisionObject()->material, groundFrictionCoeff_, 0.0, 0.01);
      }
    }
    else {
      groundFrictionCoeff_ = 0.8 + terrainCurriculumFactor_ * 0.8 * (uniDist_(gen_) - 0.5); /// c_f ~ U(0.4, 1.2)
      world_.setDefaultMaterial(groundFrictionCoeff_, 0.0, 0.01);
    }
  }

  inline void jointFrictionRandomization() {
    double jFrictionHAAHFE = 0.9 * uniDist_(gen_);  /// [0, 0.9]
    double jFrictionKFE = 1.5 * uniDist_(gen_);  /// [0, 1.5]
    jointFrictions_ << jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE,
        jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE, jFrictionHAAHFE, jFrictionHAAHFE, jFrictionKFE;
  }

  inline void simulationDtRandomization() {
    do {
      simulation_dt_ = simDtExpDist_(gen_);
    }
    while(simulation_dt_ > 0.01 || simulation_dt_ < 0.002);
    do {
      control_dt_ = conDtExpDist_(gen_);
    }
    while(control_dt_ > 0.015 || control_dt_ < 0.01);

    setSimulationTimeStep(simulation_dt_);
    setControlTimeStep(control_dt_);
  }

  inline void gainRandomization() {
    Eigen::VectorXd jointPgain;
    Eigen::VectorXd jointDgain;
    jointPgain.setZero(gvDim_);
    jointDgain.setZero(gvDim_);
    for (int i = 0; i < nJoints_; ++i) {
      jointPgain.tail(nJoints_)[i] = controller_.jointPgain_.tail(nJoints_)[i] + 2.0 * normDist_(gen_);
      jointDgain.tail(nJoints_)[i] = controller_.jointDgain_.tail(nJoints_)[i] + 0.1 * uniDist_(gen_);

      if (jointPgain.tail(nJoints_)[i] < controller_.jointPgain_.tail(nJoints_)[i] - 10.0) {
        jointPgain.tail(nJoints_)[i] = controller_.jointPgain_.tail(nJoints_)[i] - 10.0;
      } else if (jointPgain.tail(nJoints_)[i] > controller_.jointPgain_.tail(nJoints_)[i] + 10.0) {
        jointPgain.tail(nJoints_)[i] = controller_.jointPgain_.tail(nJoints_)[i] + 10.0;
      }
    }
    raibo_->setPdGains(jointPgain, jointDgain);
  }

  inline void kinematicsRandomization() {
    for (int i = 1; i < 5; i++) {
      // foot pos
      raisim::Vec<3> footCollisionOffset;
      footCollisionOffset.e() << uniDist_(gen_) * 0.01, uniDist_(gen_) * 0.01, uniDist_(gen_) * 0.02;
      footCollisionOffset[2] -= 0.24;
      raibo_->setCollisionObjectPositionOffset(2 * i, footCollisionOffset);

      // foot radius and length
      raisim::Vec<4> params;
      params[0] = 0.027 + uniDist_(gen_) * 0.006;  // radius [27, 33] mm
      params[1] = 0.025 + uniDist_(gen_) * 0.005;  // length [25, 30] mm
      raibo_->setCollisionObjectShapeParameters(2 * i, params);
    }
  }

  inline void baseInertiaParameterRandomization() {
    double randomizedMass;
    raisim::Vec<3> randomizedCoM;
    raisim::Vec<3> rpy;
    raisim::Mat<3,3> rotMat;
    Eigen::Matrix3d randomizedRotMat;
    Eigen::Vector3d randomizedD;
    raisim::Mat<3,3> randomizedI;

    /// mass randomization
    randomizedMass = nominalMass_ + (-0.5 + 1.0 * uniDist_(gen_)); /// mass sampling ~ nominal mass + U(-0.5, 0.5)

    /// com randomization
    randomizedCoM.e() << nominalCoM_[0] + (-0.05 + 0.1 * uniDist_(gen_)), nominalCoM_[1] + (-0.05 + 0.1 * uniDist_(gen_)),
                          nominalCoM_[2] + (-0.05 + 0.1 * uniDist_(gen_)); /// com sampling ~ nominal com + U(-0.05, 0.05)

    /// inertia randomization
    randomizedD << nominalD_[0] * (1 + uniDist_(gen_)), nominalD_[1] * (1 + uniDist_(gen_)),
                    nominalD_[2] * (1 + uniDist_(gen_)); /// D sampling ~ nominal D * U(1, 2)
    rpy.e() << (-5 + uniDist_(gen_) * 10.0) / 360 * M_PI, (-5 + uniDist_(gen_) * 10.0) / 360 * M_PI,
        (-5 + uniDist_(gen_) * 10.0) / 360 * M_PI; /// rpy sampling ~ U(-5, 5deg)
    raisim::rpyToRotMat_extrinsic(rpy, rotMat);
    randomizedRotMat = rotMat * nominalRot_;
    randomizedI.e() = randomizedRotMat * randomizedD.asDiagonal() * randomizedRotMat.transpose();

    auto link = raibo_->getLink("base");
    link.setWeight(randomizedMass);
    link.setComPositionInParentFrame(randomizedCoM);
    link.setInertia(randomizedI);
  }

  inline void simulateSlip() {
    for (int i = 0; i < 4; i++) {
      if (!controller_.standingMode_ && controller_.footContactState_[i]) {
        if (uniDist_(gen_) < 0.0025 || slip_[i]) {
          double footGroundFrictionCoeff = 0.1 + 0.3 * uniDist_(gen_); /// c_f ~ U(0.1, 0.4)
          world_.setMaterialPairProp(footMaterialName_[i], heightMap_->getCollisionObject()->material, footGroundFrictionCoeff, 0.0, 0.01);
          slip_[i] = true;
        }
        else {
          world_.setMaterialPairProp(footMaterialName_[i], heightMap_->getCollisionObject()->material, groundFrictionCoeff_, 0.0, 0.01);
        }
      }
      else {
        slip_[i] = false;
      }
    }
  }

  inline void clipTorque() {
    jointPos_ = raibo_->getGeneralizedCoordinate().e().tail(nJoints_);
    jointVel_ = raibo_->getGeneralizedVelocity().e().tail(nJoints_);
    double jointPgain = nominalJointPgain_, jointDgain = nominalJointDgain_;
    if (gainRandomization_) {
      jointPgain = nominalJointPgain_ +  20.0 * (-1.0 + 2.0 * uniDist_(gen_));
      jointDgain = nominalJointDgain_ +  0.2 * (-1.0 + 2.0 * uniDist_(gen_));
    }
    clippedGenForce_.tail(nJoints_) = jointPgain * (controller_.jointTarget_ - jointPos_) - jointDgain * jointVel_;

    for (int i = 0; i < nJoints_; i++) {
      /// torque - w clip
//      if (std::abs(jointVel_(i)) > jointVelClipStart_ && (std::signbit(jointVel_(i)) == std::signbit(clippedGenForce_.tail(nJoints_)(i)))) {
//        clippedTorque_ = - torqueLimit_ / jointVelClipRange_ * (std::abs(jointVel_(i)) - jointVelLimit_);
//        if (std::abs(clippedGenForce_.tail(nJoints_)(i)) > clippedTorque_) {
//          clippedGenForce_.tail(nJoints_)(i) = std::copysign(clippedTorque_, clippedGenForce_.tail(nJoints_)(i));
//        }
//      }
        /// torque limit clip
//      else {
      if (std::abs(clippedGenForce_.tail(nJoints_)(i)) > torqueLimit_) {
        clippedTorque_ = torqueLimit_;
        clippedGenForce_.tail(nJoints_)(i) = std::copysign(clippedTorque_, clippedGenForce_.tail(nJoints_)(i));
      }
//      }
    }

    if (jointFrictionRandomization_) {
      /// simulate friction
      for (int i = 0; i < nJoints_; i++) {
        frictionTorque_[6 + i] = std::copysign(jointFrictions_[i], -jointVel_[i]);
      }
      clippedGenForce_ = clippedGenForce_ + frictionTorque_;
    }

    raibo_->setGeneralizedForce(clippedGenForce_);
  }

  double step(const Eigen::Ref<EigenVec>& action, bool visualize) {
    /// action scaling
    controller_.advance(action);

    float dummy = 0.;
    int howManySteps;

    if (simulationDtRandomization_) {
      simulationDtRandomization();
    }

    if (terrainRandomization_ && groundType_ != RandomHeightMapGenerator::GroundType::SLOPE) {
      simulateSlip();
    }

    for (howManySteps = 0; howManySteps < int(control_dt_ / simulation_dt_ + 1e-10); howManySteps++) {
      subStep();
      if (visualize && visualizable_) std::this_thread::sleep_for(std::chrono::microseconds(size_t(1 * simulation_dt_ * 1e6)));

      if (isTerminalState(dummy)) {
        howManySteps++;
        break;
      }
    }

    if (uniDist_(gen_) < 1 * control_dt_ / (maxTime_ + 1e-8)) { /// 1 times command change in 1 episode
      resamplingCommand();
    }

    return controller_.getRewardSum(howManySteps);
  }

  void subStep() {
    clipTorque();
    if (server_) server_->lockVisualizationServerMutex();
    world_.integrate();
    if (server_) server_->unlockVisualizationServerMutex();
    controller_.updateStateVariables(heightMap_);
    controller_.accumulateRewards(curriculumFactor_, groundType_, terrainGenerator_.pyramidType_, terrainLevel_, iter_);

    if (uniDist_(gen_) < 0.005) {
      raibo_->getState(gc_init_from_, gv_init_from_);
      gc_init_from_[0] = 0.;
      gc_init_from_[1] = 0.;
    }
  }

  inline void resamplingCommand() {
    // command
    bool standingMode = false;
    if (command_.norm() < 1.0) {
      standingMode = normDist_(gen_) > 1.3;
    }
    controller_.setStandingMode(standingMode);
    Eigen::Vector3d newCommand;
    if (standingMode) {
      newCommand.setZero();
    } else {
      do {
        newCommand << (((maxForwardVel_ - maxForwardVelAtMaxTerrainLevel_) * std::exp(-3. * terrainLevel_) + maxForwardVelAtMaxTerrainLevel_ + 1.0) * uniDist_(gen_) - 1.0) * curriculumFactor_ * (uniDist_(gen_) < 0.3 ? 0 : 1), /// ~ U(-1.0, maxForwardVel_)
            1.0 * 2. * (uniDist_(gen_) - 0.5) * curriculumFactor_ * (uniDist_(gen_) < 0.7 ? 0 : 1), /// ~ U(-1.0, 1.0)
            1.0 * 2. * (uniDist_(gen_) - 0.5) * curriculumFactor_ * (uniDist_(gen_) < 0.4 ? 0 : 1); /// ~ U(-1.0, 1.0)
      } while((newCommand - command_).head(2).norm() > 1.0 || angleBetweenVector(newCommand.head(2), command_.head(2)) > 30.0 || std::abs(newCommand(2) - command_(2)) > 0.5 || newCommand.norm() < 0.3);
    }
    command_ << newCommand;
    controller_.setCommand(command_);
  }

  inline double angleBetweenVector(const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
    if (!a.isZero() && !b.isZero()) {
      return std::acos(a.dot(b) / (a.norm() * b.norm())) / M_PI * 180.0;
    }
    else {
      return 0;
    }
  }

  void observe(Eigen::Ref<EigenVec> ob) {
    controller_.updateObservation(observationRandomization_, gen_, normDist_);
    controller_.getObservation(obScaled_);
    ob = obScaled_.cast<float>();
  }

  void getValueObs(Eigen::Ref<EigenVec> valueOb) {
    controller_.updateValueObservation(groundTypeVector_);
    controller_.getValueObservation(valueObScaled_);
    valueOb = valueObScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) {
    if (controller_.gc_.head(2).norm() > terrainGenerator_.size / 2.0 && !testMode_) {
      return true;
    }

    return controller_.isTerminalState(terminalReward);
  }

  void setSeed(int seed) {
    gen_.seed(seed);
    terrainGenerator_.setSeed(seed);
  }

  void curriculumUpdate() {
    if (iter_ % iterPerCurriculumUpdate == 0) {
      curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);
    }
    if (iter_ % iterPerTerrainCurriculumUpdate == 0) {
      terrainCurriculumFactor_ = std::pow(terrainCurriculumFactor_, terrainCurriculumDecayFactor_);
    }
    iter_++;
  }

  void terrainChange() {
    groundType_ = RandomHeightMapGenerator::GroundType((int(groundType_) + 1 ) % terrainTypeNum_);
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    command_ = command.cast<double>();
    if (command_.norm() < 0.3) {
      controller_.setStandingMode(true);
    }
    else {
      controller_.setStandingMode(false);
    }
    controller_.setCommand(command_);
  }

  static constexpr int getObDim() { return RaiboController::getObDim(); }
  static constexpr int getValueObDim() { return RaiboController::getValueObDim(); }
  static constexpr int getActionDim() { return RaiboController::getActionDim(); }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    controller_.getState(gc, gv);
  }

  void getLoggingInfo(Eigen::Ref<EigenVec> info) {
    controller_.getLoggingInfo(info);
  }

 protected:
  int iter_ = 1;
  int iterPerCurriculumUpdate = 1;
  int iterPerTerrainCurriculumUpdate = 1;
  static constexpr int terrainTypeNum_ = 5;
  static constexpr int nJoints_ = 12;
  raisim::World world_;
  double simulation_dt_;
  double control_dt_;
  double maxTime_;
  int gcDim_, gvDim_;
  std::array<size_t, 4> footFrameIndicies_;
  const std::string resourceDir_;
  int ip=8080;

  raisim::ArticulatedSystem* raibo_;
  raisim::ArticulatedSystemVisual* raiboTarget_;
  raisim::HeightMap* heightMap_;
  Eigen::VectorXd gc_init_, gv_init_, nominalJointConfig_;
  Eigen::VectorXd gc_init_from_, gv_init_from_;
  double curriculumFactor_, curriculumDecayFactor_, maxForwardVel_;
  double terrainCurriculumFactor_, terrainCurriculumDecayFactor_, terrainLevel_ = 0.;
  double maxForwardVelAtMaxTerrainLevel_ = 2.0;
  Eigen::VectorXd obScaled_, valueObScaled_;
  bool visualizable_ = false;
  RandomHeightMapGenerator terrainGenerator_;
  RandomHeightMapGenerator::GroundType groundType_;
  Eigen::Vector3d groundTypeVector_;
  RaiboController controller_;
  Eigen::Vector3d command_;
  std::array<bool, 4> slip_;
  std::vector<std::string> footMaterialName_;
  double nominalMass_;
  raisim::Vec<3> nominalCoM_;
  raisim::Mat<3,3> nominalI_;
  Eigen::Vector3d nominalD_;
  Eigen::Matrix3d nominalRot_;
  std::vector<raisim::Vec<2>> jointLimit_;

  /// current limit: 40A -> torque limit: 10Nm, consider gear ratio: 7.15 --> torque limit: 71.5Nm
  /// joint vel limit: 48V 1400rpm (20.5046 rad/s), 10 Nm clip start at 896.3rpm (13.1273 rad/s)
  Eigen::VectorXd clippedGenForce_, frictionTorque_;
  Eigen::VectorXd jointPos_, jointVel_;
  double nominalJointPgain_, nominalJointDgain_;
  double voltage_;
  double jointVelLimit_ = 31.5;
  double jointVelClipRange_ = 20.5046 - 13.1273, jointVelClipStart_ = jointVelLimit_ - jointVelClipRange_;
  double clippedTorque_, torqueLimit_ = 71.5;

  std::unique_ptr<raisim::RaisimServer> server_;
  InstancedVisuals* scans_;

  /// previliged information for value network observation
  Eigen::VectorXd jointFrictions_;
  double groundFrictionCoeff_;

  /// randomization
  bool testMode_ = false;
  bool terrainRandomization_ = false;
  bool observationRandomization_ = false;
  bool jointFrictionRandomization_ = false;
  bool voltageRandomization_ = false;
  bool simulationDtRandomization_ = false;
  bool gainRandomization_ = false;
  bool kinematicsRandomization_ = false;
  bool baseInertiaParameterRandomization_ = false;

  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::exponential_distribution<double> simDtExpDist_;
  thread_local static std::exponential_distribution<double> conDtExpDist_;
};

thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
thread_local std::exponential_distribution<double> raisim::ENVIRONMENT::simDtExpDist_(400.);
thread_local std::exponential_distribution<double> raisim::ENVIRONMENT::conDtExpDist_(100.);
}