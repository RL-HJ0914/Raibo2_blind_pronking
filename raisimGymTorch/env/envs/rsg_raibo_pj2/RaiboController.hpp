//
// Created by donghoon on 8/11/22.
//

#ifndef _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#define _RAISIM_GYM_RAIBO_CONTROLLER_HPP
#include "RandomHeightMapGenerator.hpp"

namespace raisim {

class RaiboController {
 public:
  inline bool create(raisim::World *world) {
    raibo_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject("robot"));
    gc_.resize(raibo_->getGeneralizedCoordinateDim());
    gv_.resize(raibo_->getDOF());
    jointVelocity_.resize(nJoints_);
    previousJointVel_.resize(nJoints_);
    jointTorque_.setZero(nJoints_);
    previousTorque_.setZero(nJoints_);
    prepreviousTorque_.setZero(nJoints_);
    jointLimit_ = raibo_->getJointLimits();
    nominalBaseToFootPosXY_.setZero(8);
    baseToFootPosXY_.setZero(8);
    nominalBaseToFootPosXY_ << -0.347401, -0.19603, -0.347401, 0.19603, 0.347399, -0.19603, 0.347399, 0.19603;
    nominalJointPosWeight_.setZero(nJoints_);
    nominalJointPosWeight_ << 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0.;

    /// foot scan config
    scanConfig_.setZero(2);
    scanConfig_ << 1, 8;
    scanPoint_.resize(4, std::vector<raisim::Vec<3>>(scanConfig_.sum()));
    heightScan_.resize(4, raisim::VecDyn(scanConfig_.sum()));
    heightScan2_.resize(4, raisim::VecDyn(scanConfig_.sum()));
    scanCos_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
    scanSin_.resize(scanConfig_.size(), scanConfig_.maxCoeff());
    // precompute sin and cos because they take very long time
    for (int k = 0; k < scanConfig_.size(); k++) {
      for (int j = 0; j < scanConfig_[k]; j++) {
        const double angle = 2.0 * M_PI * double(j) / scanConfig_[k];
        scanCos_(k,j) = cos(angle);
        scanSin_(k,j) = sin(angle);
      }
    }

    /// Observation
    nominalJointConfig_.setZero(nJoints_);
    nominalJointConfig_<< 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195, 0, 0.580099, -1.195;
    jointTarget_.setZero(nJoints_);

    /// action
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    actionScaled_.setZero(actionDim_);
    previousAction_.setZero(actionDim_);
    preprevAction_.setZero(actionDim_);
    prepreprevAction_.setZero(actionDim_);

    actionMean_ << nominalJointConfig_; /// joint target
    actionStd_ << Eigen::VectorXd::Constant(nJoints_, 0.1); /// joint target

    obDouble_.setZero(obDim_);
    valueObDouble_.setZero(valueObDim_);

    /// pd controller
    jointPgain_.setZero(gvDim_); jointPgain_.tail(nJoints_).setConstant(100.0);
    jointDgain_.setZero(gvDim_); jointDgain_.tail(nJoints_).setConstant(1.0);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);

    /// indices of links that should not make contact with ground
    footIndices_.push_back(raibo_->getBodyIdx("LF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RF_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("LH_SHANK"));
    footIndices_.push_back(raibo_->getBodyIdx("RH_SHANK"));
    RSFATAL_IF(std::any_of(footIndices_.begin(), footIndices_.end(), [](int i){return i < 0;}), "footIndices_ not found")

    /// indicies of the foot frame
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RF_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("LH_S2F"));
    footFrameIndicies_.push_back(raibo_->getFrameIdxByName("RH_S2F"));
    RSFATAL_IF(std::any_of(footFrameIndicies_.begin(), footFrameIndicies_.end(), [](int i){return i < 0;}), "footFrameIndicies_ not found")

    /// exported data
    stepDataTag_ = {"command_rew",
                    "airtime_rew",
                    "base_height_rew",
                    "pronk_rew",
                    "foot_clearance_rew",
                    "foot_vel_before_contact_rew",
                    "base_motion_rew",
                    "torque_rew",
                    "joint_vel_rew",
                    "joint_accel_rew",
                    "nominal_pos_rew",
                    "joint_roll_pos_rew",
                    "smooth1_rew",
                    "smooth2_rew",
                    "GRF_smooth_rew",
                    "slip_rew",
                    "undesired_contact_rew",
                    "joint_limit_rew",
                    "joint_power_rew",
                    "undesired_GRF_rew",
                    "flight_phase_rew",
                    "torque_smooth_rew",
                    "orientation_rew",
                    "positive_rew",
                    "negative_rew"};
    stepData_.resize(stepDataTag_.size());

    return true;
  };

  void reset(std::mt19937 &gen,
             std::normal_distribution<double> &normDist) {
    raibo_->getState(gc_, gv_);
    jointTarget_ = gc_.tail(nJoints_);
    jointVelocity_ = gv_.tail(nJoints_);
    previousAction_ << gc_.tail(nJoints_);
    preprevAction_ << gc_.tail(nJoints_);
    prepreprevAction_ << gc_.tail(nJoints_);
    for (int i = 0; i < nJoints_; i++) {
      previousAction_(i) += 0.055 * normDist(gen);
    }
    for (auto & GRF : GRF_) GRF = 0;
    for (auto & GRF : undesiredGRF_) GRF = 0;
    for (auto & previousGRF : previousGRF_) previousGRF = 0;
    for (auto & prepreviousGRF : prepreviousGRF_) prepreviousGRF = 0;
    for (auto & SCF : shankContactForce_) SCF = 0;
    airTime_.setZero();
    stanceTime_.setZero();
    FootVelBeforeContact_.setZero();
    jointTorque_.setZero(nJoints_);
    previousTorque_.setZero(nJoints_);
    prepreviousTorque_.setZero(nJoints_);
  }

  void updateStateVariables(const raisim::HeightMap *map, bool initialize = false) {
    previousJointVel_ = jointVelocity_;
    raibo_->getState(gc_, gv_);
    jointVelocity_ = gv_.tail(nJoints_);

    raisim::Vec<4> quat;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, baseRot_);
    bodyLinVel_ = baseRot_.e().transpose() * gv_.segment(0, 3);
    bodyAngVel_ = baseRot_.e().transpose() * gv_.segment(3, 3);

    /// foot vel info expressed in body frame
    for (size_t i = 0; i < 4; i++) {
      raibo_->getFramePosition(footFrameIndicies_[i], globalFootPos_[i]);
      raibo_->getFrameVelocity(footFrameIndicies_[i], footVel_[i]);

      baseToFootPosXY_.segment(2 * i, 2) << (baseRot_.e().transpose() * (gc_.head(3) - globalFootPos_[i].e())).head(2);
    }

    /// height map
    controlFrameX_ =
        {baseRot_[0], baseRot_[1], 0.}; /// body x axis projected on the world x-y plane, expressed in the world frame
    controlFrameX_ /= controlFrameX_.norm();
    raisim::cross(zAxis_, controlFrameX_, controlFrameY_);
    controlRot_.e() << controlFrameX_.e(), controlFrameY_.e(), zAxis_.e();

    updateHeightScan(map);

    Eigen::Vector3d worldFrameBaseHeight;
    worldFrameBaseHeight << 0.0, 0.0, gc_(2) - map->getHeight(gc_(0), gc_(1));
    baseHeight_ = (baseRot_.e().transpose() * worldFrameBaseHeight)[2];

    /// check if the feet are in contact with the ground
    for (auto & fs : footContactState_) fs = false;
    for (auto & udfs : undesiredFootContactState_) udfs = false;
    for (auto & ss : shankContactState_) ss = false;
    for (auto & ts : thighContactState_) ts = false;

    /// get ground reaction force
    for (int i = 0; i < 4; i++) {
      prepreviousGRF_[i] = previousGRF_[i];
      previousGRF_[i] = GRF_[i];
    }
    for (auto & GRF : GRF_) GRF = 0;
    for (auto & GRF : undesiredGRF_) GRF = 0;
    for (auto & SCF : shankContactForce_) SCF = 0;
    for (auto & contactNormalAngle : contactNormalAngle_) contactNormalAngle = 0;

    for (auto & contact : raibo_->getContacts()) {
      if (contact.skip()) continue;
      auto it = std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex());
      size_t index = it - footIndices_.begin();
      if (index < 4 && !contact.isSelfCollision()) {
        if (contact.getCollisionBodyA()->name.find("FOOT") != std::string::npos || contact.getCollisionBodyB()->name.find("FOOT") != std::string::npos) {
          if (contact.isObjectA()) {
            contactNormalAngle_[index] = std::acos(contact.getNormal()[2]) * 180 / M_PI;
          }
          else {
            contactNormalAngle_[index] = std::acos(-contact.getNormal()[2]) * 180 / M_PI;
          }
          if (contactNormalAngle_[index] < 70.0) {
            footContactFrame_[index] = contact.getContactFrame().e(); /// foot contact frame is R_cw
            GRF_[index] += (contact.getImpulse().e() / (simDt_ + 1e-8)).norm();
          }
          else {
            footContactFrame_[index] = contact.getContactFrame().e(); /// foot contact frame is R_cw
            undesiredGRF_[index] += (contact.getImpulse().e() / (simDt_ + 1e-8)).norm();
          }
        }
        else {
          shankContactForce_[index] += (contact.getImpulse().e() / (simDt_ + 1e-8)).norm();
        }
      }
      else if (contact.getlocalBodyIndex() % 3 == 2) {
        thighContactState_[contact.getlocalBodyIndex() / 3] = true;
      }
    }

    /// airtime & standtime
    if (!initialize) {
      for (int i = 0; i < 4; i++) {
        if (GRF_[i] > 1.0) {
          footContactState_[i] = true;
        }
        if (undesiredGRF_[i] > 1.0) {
          footContactState_[i] = true;
          undesiredFootContactState_[i] = true;
        }
        if (shankContactForce_[i] > 1.0) {
          shankContactState_[i] = true;
        }
        if (footContactState_[i]) {
          airTime_[i] = 0;
          stanceTime_[i] += simDt_;
        } else {
          airTime_[i] += simDt_;
          stanceTime_[i] = 0;
        }
      }
      prepreviousTorque_ = previousTorque_;
      previousTorque_ = jointTorque_;
      jointTorque_ = raibo_->getGeneralizedForce().e().tail(nJoints_);
    }
  }

  void updateHeightScan(const raisim::HeightMap *map) {
    /// heightmap
    for (int k = 0; k < scanConfig_.size(); k++) {
      for (int j = 0; j < scanConfig_[k]; j++) {
        const double distance = 0.10 * k;
        for (int i = 0; i < 4; i++) {
          scanPoint_[i][scanConfig_.head(k).sum() + j][0] =
              globalFootPos_[i][0] + controlFrameX_[0] * distance * scanCos_(k,j) + controlFrameY_[0] * distance * scanSin_(k,j);
//              globalFootPos_[i][0] + footVel_[i][0] * conDt_ * j;
          scanPoint_[i][scanConfig_.head(k).sum() + j][1] =
              globalFootPos_[i][1] + controlFrameX_[1] * distance * scanCos_(k,j) + controlFrameY_[1] * distance * scanSin_(k,j);
//              globalFootPos_[i][1] + footVel_[i][1] * conDt_ * j;
          scanPoint_[i][scanConfig_.head(k).sum() + j][2] =
              map->getHeight(scanPoint_[i][scanConfig_.head(k).sum() + j][0], scanPoint_[i][scanConfig_.head(k).sum() + j][1]);

          heightScan_[i][scanConfig_.head(k).sum() + j] = (baseRot_.transpose() * (globalFootPos_[i] - scanPoint_[i][scanConfig_.head(k).sum() + j]))[2];
          heightScan2_[i][scanConfig_.head(k).sum() + j] = globalFootPos_[i][2] - scanPoint_[i][scanConfig_.head(k).sum() + j][2];
        }
      }
    }
  }

  bool advance(const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    jointTarget_ = action.cast<double>();

    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;

    pTarget_.tail(nJoints_) = jointTarget_;
    raibo_->setPdTarget(pTarget_, vTarget_);

    prepreprevAction_ = preprevAction_;
    preprevAction_ = previousAction_;
    previousAction_ = jointTarget_;

    return true;
  }

  void updateObservation(bool observationRandomization,
                         std::mt19937 &gen,
                         std::normal_distribution<double> &normDist) {
    /// body orientation
    obDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    obDouble_.segment(3, 3) = bodyAngVel_;
    /// joint pos
    obDouble_.segment(6, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    obDouble_.segment(18, nJoints_) = gv_.tail(nJoints_);
    /// previous action
    obDouble_.segment(30, nJoints_) = previousAction_;
    /// command
    obDouble_.tail(3) = command_;

    if (observationRandomization) {
      for (int i = 0; i < 3; i++) {
        obDouble_(i) += 0.05 * normDist(gen); /// orientation
        obDouble_(3 + i) += 0.22 * normDist(gen); /// body ang vel
      }
      obDouble_.head(3) /= obDouble_.head(3).norm();
      for (int i = 0; i < nJoints_; i++) {
        obDouble_(6 + i) += 0.055 * normDist(gen); /// joint pos
        obDouble_(18 + i) += 0.55 * normDist(gen); /// joint vel
      }
    }
  }

  void getObservation(Eigen::VectorXd &observation) {
    observation = obDouble_;
  }

  void updateValueObservation(const Eigen::Vector3d &groundTypeVector) {
    /// body orientation
    valueObDouble_.head(3) = baseRot_.e().row(2).transpose();
    /// body ang vel
    valueObDouble_.segment(3, 3) = bodyAngVel_;
    /// joint pos
    valueObDouble_.segment(6, nJoints_) = gc_.tail(nJoints_);
    /// joint vel
    valueObDouble_.segment(18, nJoints_) = gv_.tail(nJoints_);
    /// previous action
    valueObDouble_.segment(30, nJoints_) = previousAction_;
    /// body lin vel
    valueObDouble_.segment(42, 3) = bodyLinVel_;
    /// air time, stance time
    valueObDouble_.segment(45, 8) << airTime_, stanceTime_;
    /// base height
    valueObDouble_.segment(53, 1) << baseHeight_;
    /// GRF
    valueObDouble_.segment(54, 4) << GRF_[0], GRF_[1], GRF_[2], GRF_[3];
    /// previous GRF
    valueObDouble_.segment(58, 4) << previousGRF_[0], previousGRF_[1], previousGRF_[2], previousGRF_[3];
    /// foot contact, undesired contact state and undesired GRF
    valueObDouble_.segment(62, 4) << FootVelBeforeContact_;
    valueObDouble_.segment(66, 16) << footContactState_[0], footContactState_[1], footContactState_[2], footContactState_[3],
                                      undesiredFootContactState_[0], undesiredFootContactState_[1], undesiredFootContactState_[2], undesiredFootContactState_[3],
                                      undesiredGRF_[0], undesiredGRF_[1],undesiredGRF_[2],undesiredGRF_[3],
                                      shankContactState_[0], shankContactState_[1], shankContactState_[2], shankContactState_[3];
    /// contact normal angle
    valueObDouble_.segment(82, 4) << contactNormalAngle_[0], contactNormalAngle_[1], contactNormalAngle_[2], contactNormalAngle_[3];
    /// ground type
    valueObDouble_.segment(86, 3) << groundTypeVector;
    /// height scan
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < scanConfig_.sum(); j++)
        valueObDouble_[89 + i * scanConfig_.sum() + j] = heightScan2_[i][j];
    /// command
    valueObDouble_.tail(3) = command_;
  }

  void getValueObservation(Eigen::VectorXd &valueObservation) {
    valueObservation = valueObDouble_;
  }

  inline void setRewardConfig(const Yaml::Node &cfg) {
    READ_YAML(double, commandTrackingRewardCoeff, cfg["reward"]["command_tracking_reward_coeff"])
    READ_YAML(double, airtimeRewardCoeff_, cfg["reward"]["airtime_reward_coeff"])
    READ_YAML(double, baseHeightRewardCoeff_, cfg["reward"]["base_height_reward_coeff"])
    READ_YAML(double, footClearanceRewardCoeff_, cfg["reward"]["foot_clearance_reward_coeff"])
    READ_YAML(double, footVelBeforeContactRewardCoeff_, cfg["reward"]["foot_vel_before_contact_reward_coeff"])
    READ_YAML(double, baseMotionRewardCoeff_, cfg["reward"]["base_motion_reward_coeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["reward"]["torque_reward_coeff"])
    READ_YAML(double, jointVelocityRewardCoeff_, cfg["reward"]["joint_velocity_reward_coeff"])
    READ_YAML(double, jointAccelerationRewardCoeff_, cfg["reward"]["joint_accel_reward_coeff"])
    READ_YAML(double, nominalPosRewardCoeff_, cfg["reward"]["nominal_pos_reward_coeff"])
    READ_YAML(double, jointRollPosRewardCoeff_, cfg["reward"]["joint_roll_pos_reward_coeff"])
    READ_YAML(double, smoothReward1Coeff_, cfg["reward"]["smooth_reward1_coeff"])
    READ_YAML(double, smoothReward2Coeff_, cfg["reward"]["smooth_reward2_coeff"])
    READ_YAML(double, GRFsmoothRewardCoeff_, cfg["reward"]["GRF_smooth_reward_coeff"])
    READ_YAML(double, torqueSmoothRewardCoeff_, cfg["reward"]["torque_smooth_reward_coeff"])
    READ_YAML(double, orientationRewardCoeff_, cfg["reward"]["orientation_reward_coeff"])
    READ_YAML(double, slipRewardCoeff_, cfg["reward"]["slip_reward_coeff"])
    READ_YAML(double, undesiredContactRewardCoeff_, cfg["reward"]["undesired_contact_reward_coeff"])
    READ_YAML(double, jointLimitRewardCoeff_, cfg["reward"]["joint_limit_reward_coeff"])
    READ_YAML(double, jointPowerRewardCoeff_, cfg["reward"]["joint_power_reward_coeff"])
    READ_YAML(double, undesiredGRFRewardCoeff_, cfg["reward"]["undesired_GRF_reward_coeff"])
    READ_YAML(double, flightPhaseRewardCoeff_, cfg["reward"]["flight_phase_reward_coeff"])
    READ_YAML(double, pronkRewardCoeff_, cfg["reward"]["pronk_reward_coeff"])
    READ_YAML(double, smoothRewardCurriculumEnd_, cfg["curriculum"]["target_smoothness_end"])
  }

  inline void accumulateRewards(const double &cf, const RandomHeightMapGenerator::GroundType &groundType, const bool &pyramidOrNot, const double &envTerrainLevel, const int &iter) {
    double targetBaseHeight = 0.53;
    double targetFootHeight = 0.08;
    double terrainLevel = envTerrainLevel;
    double linearCommandTrackingReward = 0., angularCommandTrackingReward = 0.;
    linearCommandTrackingReward += std::exp(-1.0 * (command_.head(2) - bodyLinVel_.head(2)).squaredNorm());
    linearCommandTrackingReward *= (1.0 + std::exp(-0.5 * (command_.head(2) - bodyLinVel_.head(2)).norm()));
    angularCommandTrackingReward += std::exp(-1.5 * pow((command_(2) - bodyAngVel_(2)), 2));
    commandTrackingReward_ += (linearCommandTrackingReward + angularCommandTrackingReward) * commandTrackingRewardCoeff;

    baseHeightReward_ += baseHeightRewardCoeff_ * std::exp(-10.0 * std::abs(targetBaseHeight - baseHeight_));

    baseMotionReward_ += baseMotionRewardCoeff_ * (0.02 * fabs(bodyAngVel_[0]) + 0.02 * fabs(bodyAngVel_[1]));

    torqueReward_ += torqueRewardCoeff_ * jointTorque_.squaredNorm();

    jointVelocityReward_ += cf * jointVelocityRewardCoeff_ * jointVelocity_.squaredNorm();

    jointAccelerationReward_ += cf * jointAccelerationRewardCoeff_ * (gv_.tail(12) - previousJointVel_).squaredNorm();

    double nominalPosReward = 0., orientationReward = 0.;
    orientationReward += cf * orientationRewardCoeff_ * std::acos(baseRot_(8)) * std::acos(baseRot_(8));
    if (standingMode_) {
      if (groundType == RandomHeightMapGenerator::GroundType::TUK || pyramidOrNot) {
        terrainLevel = 0.;
      }
      nominalPosReward += cf * nominalPosRewardCoeff_ * (1.0 / (1 + 3 * terrainLevel)) * (gc_.tail(nJoints_) - nominalJointConfig_).norm();
      nominalPosReward *= 4.0;
      orientationReward *= 10.0;
    }
    else {
      nominalPosReward += cf * nominalPosRewardCoeff_ * (nominalBaseToFootPosXY_ - baseToFootPosXY_).norm();
    }
    nominalPosReward_ += nominalPosReward;
    orientationReward_ += orientationReward;

    smoothReward1_ += cf * smoothReward1Coeff_ * (previousAction_ - preprevAction_).squaredNorm() * std::min(iter, int(smoothRewardCurriculumEnd_)) / smoothRewardCurriculumEnd_;
    smoothReward2_ += cf * smoothReward2Coeff_ * (prepreprevAction_ + previousAction_ - 2 * preprevAction_).squaredNorm() * std::min(iter, int(smoothRewardCurriculumEnd_)) / smoothRewardCurriculumEnd_;
    torqueSmoothReward_ += cf * torqueSmoothRewardCoeff_ * (0.5 * (prepreviousTorque_ + jointTorque_ - 2 * previousTorque_).squaredNorm() + (jointTorque_ - previousTorque_).squaredNorm());

    double undesiredContactNum = 0.;
    for (int i = 0; i < 4; i++) {
      GRFsmoothReward_ += cf * GRFsmoothRewardCoeff_ * (0.5 * std::pow(prepreviousGRF_[i] + GRF_[i] - 2 * previousGRF_[i], 2) + std::pow(previousGRF_[i] - GRF_[i], 2));

      undesiredGRFReward_ += cf * undesiredGRFRewardCoeff_ * (undesiredGRF_[i] + shankContactForce_[i]);

      undesiredContactNum += undesiredFootContactState_[i] + shankContactState_[i];

      if (standingMode_) {
        airtimeReward_ += std::min(std::max(stanceTime_[i] - airTime_[i], -0.25), 0.25) * airtimeRewardCoeff_;
        jointRollPosReward_ += cf * jointRollPosRewardCoeff_ * pow(gc_.tail(nJoints_)(3 * i), 2);
      }
      else {
        if (airTime_[i] < 0.25)
          airtimeReward_ += std::min(airTime_[i], 0.2) * airtimeRewardCoeff_;
        if (stanceTime_[i] < 0.2)
          airtimeReward_ += 3*std::min(stanceTime_[i], 0.2) * airtimeRewardCoeff_;

        jointRollPosReward_ += cf * jointRollPosRewardCoeff_ * (std::abs(command_[0]) / (command_.norm() + 1e-8)) * pow(gc_.tail(nJoints_)(3 * i), 2);
      }

      if (footContactState_[i]) {
        slipReward_ += cf * slipRewardCoeff_ * (footContactFrame_[i] * footVel_[i].e()).head(2).squaredNorm();
        footVelBeforeContactReward_ += footVelBeforeContactRewardCoeff_ * FootVelBeforeContact_[i];
      }
      else {
        if (airTime_[i] > 0.05) {
          FootVelBeforeContact_[i] = footVel_[i].e().norm();
        }
        if (!standingMode_) {
          if (groundType == RandomHeightMapGenerator::GroundType::SLOPE) {
              footClearanceReward_ += footClearanceRewardCoeff_ *
                  std::pow(heightScan_[i][0] - targetFootHeight, 2) *
                  footVel_[i].e().norm() / (0.2 + command_.norm());
            }
          else if (groundType == RandomHeightMapGenerator::GroundType::HEIGHT_MAP) {
            footClearanceReward_ += footClearanceRewardCoeff_ *
                std::pow(heightScan2_[i][0] - targetFootHeight, 2) *
                footVel_[i].e().norm() / (0.2 + command_.norm());
          }
          else {
            if (heightScan2_[i][0] < targetFootHeight) {
              footClearanceReward_ += footClearanceRewardCoeff_ *
                  std::pow(heightScan2_[i][0] - targetFootHeight, 2) *
                  footVel_[i].e().norm() / (0.2 + command_.head(2).norm());
            }
          }
        }
      }
    }
    undesiredContactReward_ += cf * undesiredContactRewardCoeff_ * undesiredContactNum;

    if (!footContactState_[0] && !footContactState_[1] && !footContactState_[2] && !footContactState_[3]) {
      flightPhaseReward_ += cf * flightPhaseRewardCoeff_;
    }

    double jointLimitReward = 0.;
    double motorTorque = 0., current = 0.;
    for (int i = 0; i < nJoints_; i++) {
      motorTorque = std::abs(jointTorque_[i]) / 7.15;
      if (motorTorque < 4) {
        current = motorTorque / 0.281;
      }
      else {
        current = 1.53 + 2.65 * motorTorque + 0.13 * motorTorque * motorTorque;
      }
//      jointPowerReward_ += cf * jointPowerRewardCoeff_ * (gv_.tail(nJoints_)[i] * jointTorque_[i] + current * current * (0.104 + 0.0148) + current * 0.21);
      jointPowerReward_ += cf * jointPowerRewardCoeff_ * (current * current * (0.104 + 0.0148) + current * 0.21);

      if (i % 3 == 2) {
        if (gc_[i + 7] > jointLimit_[i + 6][0] && gc_[i + 7] < jointLimit_[i + 6][1]) {
          jointLimitReward += (std::pow(std::log10(std::abs(gc_[i + 7] - jointLimit_[i + 6][0])), 2) + std::pow(std::log10(std::abs(gc_[i + 7] - jointLimit_[i + 6][1])), 2)) / 10;
        }
      }
    }
    jointLimitReward_ += cf * jointLimitRewardCoeff_ * jointLimitReward;

    /// Pronk reward
    if ((!footContactState_[0] && !footContactState_[1] && !footContactState_[2] && !footContactState_[3]) ||
        (footContactState_[0] && footContactState_[1] && footContactState_[2] && footContactState_[3]))
    {
      if (!standingMode_) pronkReward_ += pronkRewardCoeff_ ;
    }
  }

  [[nodiscard]] float getRewardSum(const int & howManySteps) {
    int positiveRewardNum = 4;
    double positiveReward, negativeReward;
    stepData_[0] = commandTrackingReward_;
    stepData_[1] = airtimeReward_;
    stepData_[2] = baseHeightReward_;
    stepData_[3] = pronkReward_;
    stepData_[4] = footClearanceReward_;
    stepData_[5] = footVelBeforeContactReward_;
    stepData_[6] = baseMotionReward_;
    stepData_[7] = torqueReward_;
    stepData_[8] = jointVelocityReward_;
    stepData_[9] = jointAccelerationReward_;
    stepData_[10] = nominalPosReward_;
    stepData_[11] = jointRollPosReward_;
    stepData_[12] = smoothReward1_;
    stepData_[13] = smoothReward2_;
    stepData_[14] = GRFsmoothReward_;
    stepData_[15] = slipReward_;
    stepData_[16] = undesiredContactReward_;
    stepData_[17] = jointLimitReward_;
    stepData_[18] = jointPowerReward_;
    stepData_[19] = undesiredGRFReward_;
    stepData_[20] = flightPhaseReward_;
    stepData_[21] = torqueSmoothReward_;
    stepData_[22] = orientationReward_;
    stepData_ /= howManySteps;
    positiveReward = stepData_.head(positiveRewardNum).sum();
    negativeReward = stepData_.segment(positiveRewardNum, stepData_.size() - positiveRewardNum - 2).sum();
    stepData_[23] = positiveReward;
    stepData_[24] = negativeReward;

    commandTrackingReward_ = 0.;
    airtimeReward_ = 0.;
    baseHeightReward_ = 0.;
    pronkReward_ = 0.;
    footClearanceReward_ = 0.;
    footVelBeforeContactReward_ = 0.;
    baseMotionReward_ = 0.;
    torqueReward_ = 0.;
    jointVelocityReward_ = 0.;
    jointAccelerationReward_ = 0.;
    nominalPosReward_ = 0.;
    jointRollPosReward_ = 0.;
    smoothReward1_ = 0.;
    smoothReward2_ = 0.;
    GRFsmoothReward_ = 0.;
    slipReward_ = 0.;
    undesiredContactReward_ = 0.;
    jointLimitReward_ = 0.;
    jointPowerReward_ = 0.;
    undesiredGRFReward_ = 0.;
    flightPhaseReward_ = 0.;
    torqueSmoothReward_ = 0.;
    orientationReward_ = 0.;

    return float(positiveReward * std::exp(0.1 * negativeReward));
  }

  [[nodiscard]] bool isTerminalState(float &terminalReward) {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not a foot or shank
    for (auto & contact : raibo_->getContacts())
      if (std::find(footIndices_.begin(), footIndices_.end(), contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    for (int i = 0; i < nJoints_; i++) {
      if (gc_[i + 7] < jointLimit_[i + 6][0]) {
        return true;
      }
      if (gc_[i + 7] > jointLimit_[i + 6][1]) {
        return true;
      }
    }

    terminalReward = 0.f;
    return false;
  }

  inline void setStandingMode(bool mode) { standingMode_ = mode; }
  inline void setCommand(const Eigen::Vector3d &command) { command_ = command; }

  [[nodiscard]] static constexpr int getObDim() { return obDim_; }
  [[nodiscard]] static constexpr int getValueObDim() { return valueObDim_; }
  [[nodiscard]] static constexpr int getActionDim() { return actionDim_; }
  [[nodiscard]] double getSimDt() { return simDt_; }
  [[nodiscard]] double getConDt() { return conDt_; }
  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) { gc = gc_.cast<float>(); gv = gv_.cast<float>(); }

  void getLoggingInfo(Eigen::Ref<EigenVec> info) {
    Eigen::VectorXd infoBag;
    infoBag.setZero(62);
    std::array<raisim::Vec<3>, 4> bodyFrameFootVel;
    Eigen::Vector4d footStride;
    for (int i = 0; i < 4; i++) {
      bodyFrameFootVel[i].e() = baseRot_.e().transpose() * footVel_[i].e();
      footStride[i] = ((nominalBaseToFootPosXY_ - baseToFootPosXY_).segment(2 * i, 2)).norm();
    }
    infoBag << bodyFrameFootVel[0].e().head(2).norm(), bodyFrameFootVel[1].e().head(2).norm(), bodyFrameFootVel[2].e().head(2).norm(), bodyFrameFootVel[3].e().head(2).norm(),
        bodyFrameFootVel[0].e().norm(), bodyFrameFootVel[1].e().norm(), bodyFrameFootVel[2].e().norm(), bodyFrameFootVel[3].e().norm(),
        bodyFrameFootVel[0].e()[2], bodyFrameFootVel[1].e()[2], bodyFrameFootVel[2].e()[2], bodyFrameFootVel[3].e()[2],
        raibo_->getGeneralizedForce().e().tail(nJoints_),
        airTime_, stanceTime_,
        footContactState_[0], footContactState_[1], footContactState_[2], footContactState_[3],
        globalFootPos_[0](2), globalFootPos_[1](2), globalFootPos_[2](2), globalFootPos_[3](2),
        command_, bodyLinVel_.head(2), bodyAngVel_.tail(1),
        GRF_[0], GRF_[1], GRF_[2], GRF_[3],
        undesiredFootContactState_[0], undesiredFootContactState_[1], undesiredFootContactState_[2], undesiredFootContactState_[3],
        undesiredGRF_[0], undesiredGRF_[1], undesiredGRF_[2], undesiredGRF_[3],
        footStride;
    info = infoBag.cast<float>();
  }

  void setSimDt(double dt) { simDt_ = dt; };
  void setConDt(double dt) { conDt_ = dt; };

  [[nodiscard]] inline const std::vector<std::string> &getStepDataTag() const { return stepDataTag_; }
  [[nodiscard]] inline const Eigen::VectorXd &getStepData() const { return stepData_; }

  // robot configuration variables
  raisim::ArticulatedSystem *raibo_;
  std::vector<size_t> footIndices_, footFrameIndicies_;
  Eigen::VectorXd nominalJointConfig_;
  static constexpr int nJoints_ = 12;
  static constexpr int actionDim_ = 12;
  static constexpr size_t obDim_ = 45;
  static constexpr size_t valueObDim_ = 92 + 9 * 4;
  double simDt_ = .0025;
  static constexpr int gcDim_ = 19;
  static constexpr int gvDim_ = 18;

  // robot state variables
  Eigen::VectorXd gc_, gv_;
  Eigen::Vector3d bodyLinVel_, bodyAngVel_; /// body velocities are expressed in the body frame
  Eigen::VectorXd jointVelocity_, previousJointVel_;
  std::array<raisim::Vec<3>, 4> globalFootPos_, footVel_;
  Eigen::VectorXd nominalBaseToFootPosXY_, baseToFootPosXY_;
  raisim::Vec<3> zAxis_ = {0., 0., 1.}, controlFrameX_, controlFrameY_;
  std::array<bool, 4> footContactState_, undesiredFootContactState_, shankContactState_, thighContactState_;
  std::array<Eigen::Matrix3d, 4> footContactFrame_;
  raisim::Mat<3, 3> baseRot_, controlRot_;
  Eigen::Vector4d airTime_, stanceTime_;
  std::vector<raisim::Vec<2>> jointLimit_;
  Eigen::Vector4d FootVelBeforeContact_;

  // robot observation variables
  Eigen::VectorXd obDouble_;
  Eigen::VectorXd valueObDouble_;
  std::array<double, 4> GRF_, previousGRF_, prepreviousGRF_, undesiredGRF_, shankContactForce_, contactNormalAngle_;
  std::vector<raisim::VecDyn> heightScan_, heightScan2_;
  Eigen::VectorXi scanConfig_;
  std::vector<std::vector<raisim::Vec<3>>> scanPoint_;
  Eigen::MatrixXd scanSin_;
  Eigen::MatrixXd scanCos_;
  Eigen::Vector3d command_;

  // control variables
  double conDt_ = 0.01;
  bool standingMode_ = false;
  Eigen::VectorXd actionMean_, actionStd_, actionScaled_, previousAction_, preprevAction_, prepreprevAction_;
  Eigen::VectorXd jointTorque_, previousTorque_, prepreviousTorque_;
  Eigen::VectorXd pTarget_, vTarget_; // full robot gc dim
  Eigen::VectorXd jointTarget_;
  Eigen::VectorXd jointPgain_, jointDgain_;

  // reward variables
  double commandTrackingRewardCoeff = 0., commandTrackingReward_ = 0.;
  double baseMotionRewardCoeff_ = 0., baseMotionReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  double smoothReward1Coeff_ = 0., smoothReward1_ = 0.;
  double smoothReward2Coeff_ = 0., smoothReward2_ = 0.;
  double GRFsmoothRewardCoeff_ = 0., GRFsmoothReward_ = 0.;
  double jointVelocityRewardCoeff_ = 0., jointVelocityReward_ = 0.;
  double jointAccelerationRewardCoeff_ = 0., jointAccelerationReward_ = 0.;
  double nominalPosRewardCoeff_ = 0., nominalPosReward_ = 0.;
  double jointRollPosRewardCoeff_ = 0., jointRollPosReward_ = 0.;
  double footClearanceRewardCoeff_ = 0., footClearanceReward_ = 0.;
  double footVelBeforeContactRewardCoeff_ = 0., footVelBeforeContactReward_ = 0.;
  double slipRewardCoeff_ = 0., slipReward_ = 0.;
  double airtimeRewardCoeff_ = 0., airtimeReward_ = 0.;
  double baseHeightRewardCoeff_ = 0., baseHeightReward_ = 0.;
  double undesiredContactRewardCoeff_ = 0., undesiredContactReward_ = 0.;
  double jointLimitRewardCoeff_ = 0., jointLimitReward_ = 0.;
  double jointPowerRewardCoeff_ = 0., jointPowerReward_ = 0.;
  double undesiredGRFRewardCoeff_ = 0., undesiredGRFReward_ = 0.;
  double flightPhaseRewardCoeff_ = 0., flightPhaseReward_ = 0.;
  double torqueSmoothRewardCoeff_ = 0., torqueSmoothReward_ = 0.;
  double orientationRewardCoeff_ = 0., orientationReward_ = 0.;
  double pronkRewardCoeff_ = 0., pronkReward_ = 0.;
  double terminalRewardCoeff_ = 0.0;
  double baseHeight_;
  Eigen::VectorXd nominalJointPosWeight_;
  double smoothRewardCurriculumEnd_;

  // exported data
  Eigen::VectorXd stepData_;
  std::vector<std::string> stepDataTag_;
};

}

#endif //_RAISIM_GYM_RAIBO_CONTROLLER_HPP