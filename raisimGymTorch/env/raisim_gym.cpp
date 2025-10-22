//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

int THREAD_COUNT = 1;

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, "RaisimGymRaiboPJ2")
    .def(py::init<std::string, std::string>())
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("getValueObs", &VectorizedEnvironment<ENVIRONMENT>::getValueObs)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("step_visualize", &VectorizedEnvironment<ENVIRONMENT>::step_visualize)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getValueObDim", &VectorizedEnvironment<ENVIRONMENT>::getValueObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ENVIRONMENT>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("terrainChange", &VectorizedEnvironment<ENVIRONMENT>::terrainChange)
    .def("curriculumUpdate", &VectorizedEnvironment<ENVIRONMENT>::curriculumUpdate)
    .def("getStepDataTag", &VectorizedEnvironment<ENVIRONMENT>::getStepDataTag)
    .def("getStepData", &VectorizedEnvironment<ENVIRONMENT>::getStepData)
    .def("setCommand", &VectorizedEnvironment<ENVIRONMENT>::setCommand)
    .def("getState", &VectorizedEnvironment<ENVIRONMENT>::getState)
    .def("getLoggingInfo", &VectorizedEnvironment<ENVIRONMENT>::getLoggingInfo)
    .def("getObStatistics", &VectorizedEnvironment<ENVIRONMENT>::getObStatistics)
    .def("setObStatistics", &VectorizedEnvironment<ENVIRONMENT>::setObStatistics)
    .def("getValueObStatistics", &VectorizedEnvironment<ENVIRONMENT>::getValueObStatistics)
    .def("setValueObStatistics", &VectorizedEnvironment<ENVIRONMENT>::setValueObStatistics);

  py::class_<NormalSampler>(m, "NormalSampler")
      .def(py::init<int>(), py::arg("dim"))
      .def("seed", &NormalSampler::seed)
      .def("sample", &NormalSampler::sample);
}
