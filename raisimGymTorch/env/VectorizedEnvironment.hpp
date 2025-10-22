//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Core>
#include "BasicEigenTypes.hpp"
extern int THREAD_COUNT;

namespace raisim {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg)
      : resourceDir_(resourceDir) {
    Yaml::Parse(cfg_, cfg);
    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();
    double simDt, conDt;
    READ_YAML(double, simDt, cfg_["simulation_dt"])
    READ_YAML(double, conDt, cfg_["control_dt"])

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0, i));
      environments_.back()->setSimulationTimeStep(simDt);
      environments_.back()->setControlTimeStep(conDt);
    }

    int startSeed;
    READ_YAML(int, startSeed, cfg_["seed"])
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->setSeed(startSeed + i);
      environments_[i]->init();
      environments_[i]->reset();
    }

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(getObDim());
      obVar_.setOnes(getObDim());
      recentMean_.setZero(getObDim());
      recentVar_.setZero(getObDim());
      delta_.setZero(getObDim());
      epsilon_.setZero(getObDim());
      epsilon_.setConstant(1e-8);

      valueObMean_.setZero(getValueObDim());
      valueObVar_.setOnes(getValueObDim());
      valueRecentMean_.setZero(getValueObDim());
      valueRecentVar_.setZero(getValueObDim());
      valueDelta_.setZero(getValueObDim());
      valueEpsilon_.setZero(getValueObDim());
      valueEpsilon_.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics=false) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  void getValueObs(Eigen::Ref<EigenRowMajorMat> &valueOb, bool updateStatistics=false) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->getValueObs(valueOb.row(i));

    if (normalizeObservation_)
      updateValueObservationStatisticsAndNormalize(valueOb, updateStatistics);
  }

  std::vector<std::string> getStepDataTag() {
    return environments_[0]->getStepDataTag();
  }

  int getStepData(int sample_size,
                  Eigen::Ref<EigenDoubleVec> &mean,
                  Eigen::Ref<EigenDoubleVec> &squareSum,
                  Eigen::Ref<EigenDoubleVec> &min,
                  Eigen::Ref<EigenDoubleVec> &max) {
    size_t data_size = getStepDataTag().size();
    if( data_size == 0 ) return sample_size;

    RSFATAL_IF(mean.size() != data_size ||
        squareSum.size() != data_size ||
        min.size() != data_size ||
        max.size() != data_size, "vector size mismatch")

    mean *= sample_size;

    for (int i = 0; i < num_envs_; i++) {
      mean += environments_[i]->getStepData();
      for (int j = 0; j < data_size; j++) {
        min(j) = std::min(min(j), environments_[i]->getStepData()[j]);
        max(j) = std::max(max(j), environments_[i]->getStepData()[j]);
      }
    }

    sample_size += num_envs_;
    mean /= sample_size;
    for (int i = 0; i < num_envs_; i++) {
      for (int j = 0; j < data_size; j++) {
        double temp = environments_[i]->getStepData()[j];
        squareSum[j] += temp * temp;
      }
    }

    return sample_size;
  }

  void getState(Eigen::Ref<EigenVec> gc, Eigen::Ref<EigenVec> gv) {
    environments_[0]->getState(gc, gv);
  }

  void getLoggingInfo(Eigen::Ref<EigenVec> info) {
    environments_[0]->getLoggingInfo(info);
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done, false);
  }

  void step_visualize(Eigen::Ref<EigenRowMajorMat> &action,
                      Eigen::Ref<EigenVec> &reward,
                      Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done, true);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }
  void getValueObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var) {
    mean = valueObMean_; var = valueObVar_; }
  void setValueObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    valueObMean_ = mean; valueObVar_ = var; valueObCount_ = count; }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void setCommand(const Eigen::Ref<EigenVec>& command) {
    environments_[0]->setCommand(command);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return ChildEnvironment::getObDim(); }
  int getValueObDim() { return ChildEnvironment::getValueObDim(); }
  int getActionDim() { return ChildEnvironment::getActionDim(); }
  int getNumOfEnvs() { return num_envs_; }
  ////// optional methods //////
  void terrainChange() {
    for (auto *env: environments_)
      env->terrainChange();
  }

  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

 private:

  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for (int i = 0; i < getObDim(); i++)
        delta_[i] = delta_[i] * delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient((obVar_ + epsilon_).cwiseSqrt().transpose());
  }

  void updateValueObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      valueRecentMean_ = ob.colwise().mean();
      valueRecentVar_ = (ob.rowwise() - valueRecentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      valueDelta_ = valueObMean_ - valueRecentMean_;
      for (int i = 0; i < getValueObDim(); i++)
        valueDelta_[i] = valueDelta_[i] * valueDelta_[i];

      float totCount = valueObCount_ + num_envs_;

      valueObMean_ = valueObMean_ * (valueObCount_ / totCount) + valueRecentMean_ * (num_envs_ / totCount);
      valueObVar_ = (valueObVar_ * valueObCount_ + valueRecentVar_ * num_envs_ + valueDelta_ * (valueObCount_ * num_envs_ / totCount)) / (totCount);
      valueObCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i = 0; i < num_envs_; i++)
      ob.row(i) = (ob.row(i) - valueObMean_.transpose()).template cwiseQuotient((valueObVar_ + valueEpsilon_).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done,
                           bool visualize) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId), visualize);

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  std::vector<ChildEnvironment *> environments_;

  int num_envs_ = 1;
  bool render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon_;

  EigenVec valueObMean_;
  EigenVec valueObVar_;
  float valueObCount_ = 1e-4;
  EigenVec valueRecentMean_, valueRecentVar_, valueDelta_;
  EigenVec valueEpsilon_;
};

class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[i].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }

  int dim_;
  std::vector<NormalDistribution> normal_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP