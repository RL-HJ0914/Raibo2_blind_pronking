// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#ifndef _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_
#define _RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_

#include "raisim/World.hpp"

namespace raisim {

class RandomHeightMapGenerator {
 public:

  enum class GroundType : int {
    HEIGHT_MAP = 0,
    SLOPE,
    STAIRS,
    STEPS,
    NOSING_STAIRS,
    TUK,
    HEIGHT_MAP_DISCRETE,
    STEPS_INCLINE,
    STAIRS3,
    FLAT
  };

  RandomHeightMapGenerator() = default;

  void setSeed(int seed) {
    terrain_seed_ = seed;
  }

  bool setTerrainParams(const GroundType &groundType,
                        const Eigen::Vector4d &terrainParams) {
    switch (groundType) {
      case GroundType::HEIGHT_MAP:
        gridSize = 0.2;
        size = terrainParams[0];
        nGrid = (int) (size / gridSize) + 1;
        size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation
        terrainProperties_.frequency = terrainParams[1];
        terrainProperties_.zScale = terrainParams[2];
        terrainProperties_.xSize = size;
        terrainProperties_.ySize = size;
        terrainProperties_.xSamples = nGrid;
        terrainProperties_.ySamples = nGrid;
        terrainProperties_.fractalOctaves = 5;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_;
        terrain_seed_ += 500;
        terrainProperties_.stepSize = 0.;
        return true;

      case GroundType::HEIGHT_MAP_DISCRETE:
        gridSize = 0.2;
        size = terrainParams[0];
        nGrid = (int) (size / gridSize) + 1;
        size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation
        terrainProperties_.frequency = 0.3;
        terrainProperties_.zScale = terrainParams[1];  // curriculumFactor * 1.2;
        terrainProperties_.xSize = size;
        terrainProperties_.ySize = size;
        terrainProperties_.xSamples = nGrid;
        terrainProperties_.ySamples = nGrid;
        terrainProperties_.fractalOctaves = 3;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_;
        terrain_seed_ += 500;
        terrainProperties_.stepSize = terrainParams[2]; // 0.1 * curriculumFactor;
        return true;

      case GroundType::STEPS:
        gridSize = 0.02;
        size = terrainParams[0];
        stepWidth = terrainParams[1];
        stepHeight = terrainParams[2];
        nGrid = (int) (size / gridSize);
        nBlock = (int) (size / stepWidth);
        nGridPerBlock = nGrid / nBlock;
        nGrid = nGridPerBlock * nBlock;  // remove blank part due to (int) operation
        size = nGrid * gridSize;  // remove blank part due to (int) operation
        return true;

      case GroundType::STEPS_INCLINE:
        gridSize = 0.1;
        size = terrainParams[0];
        roughness = terrainParams[1];
        stepHeight = terrainParams[2];
        nGrid = (int) (size / gridSize);
        nBlock = 10;
        nGridPerBlock = nGrid / nBlock;
        nGrid = nGridPerBlock * nBlock;  // remove blank part due to (int) operation
        size = nGrid * gridSize;  // remove blank part due to (int) operation
        return true;

      case GroundType::STAIRS:
        gridSize = 0.02;
        size = terrainParams[0];
        stepWidth = terrainParams[1];
        stepHeight = terrainParams[2];
        slopeAngle = terrainParams[3];
        return true;

      case GroundType::NOSING_STAIRS:
        gridSize = 0.02;
        size = terrainParams[0];
        stepWidth = terrainParams[1];
        stepHeight = terrainParams[2];
        slopeAngle = terrainParams[3];
        return true;

      case GroundType::SLOPE:
        gridSize = 0.2;
        size = terrainParams[0];
        terrainProperties_.frequency = terrainParams[1];
        terrainProperties_.zScale = terrainParams[2];
        slopeAngle = terrainParams[3];
        nGrid = (int) (size / gridSize) + 1;
        size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation
        terrainProperties_.xSize = size;
        terrainProperties_.ySize = size;
        terrainProperties_.xSamples = nGrid;
        terrainProperties_.ySamples = nGrid;
        terrainProperties_.fractalOctaves = 5;
        terrainProperties_.fractalLacunarity = 3.0;
        terrainProperties_.fractalGain = 0.45;
        terrainProperties_.seed = terrain_seed_;
        terrain_seed_ += 500;
        terrainProperties_.stepSize = 0.;
        return true;

      case GroundType::STAIRS3:
        gridSize = 0.02;
        size = terrainParams[0];
        stepWidth = terrainParams[1];
        stepHeight = terrainParams[2];
        slopeAngle = terrainParams[3];
        return true;

      case GroundType::TUK:
        gridSize = 0.2;
        size = terrainParams[0];
        stepHeight = terrainParams[1];
        return true;

      case GroundType::FLAT:
        size = 20.0;
        return true;
    }
    return true;
  }

  raisim::HeightMap *generateTerrain(raisim::World *world,
                                     const GroundType &groundType,
                                     std::mt19937 &gen,
                                     std::uniform_real_distribution<double> &uniDist) {
    std::vector<double> heightVec;
    std::unique_ptr<raisim::TerrainGenerator> genPtr;
    normalVector_.setZero();

    switch (groundType) {
      case GroundType::HEIGHT_MAP:
        terrainProperties_.fractalLacunarity = 1.0 + 4.0 * uniDist(gen);
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);

      case GroundType::HEIGHT_MAP_DISCRETE:
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);

      case GroundType::STEPS:
        heightVec.resize(nGrid * nGrid);
        for (int xBlock = 0; xBlock < nBlock; xBlock++) {
          for (int yBlock = 0; yBlock < nBlock; yBlock++) {
            double height = stepHeight * uniDist(gen);
            for (int i = 0; i < nGridPerBlock; i++) {
              for (int j = 0; j < nGridPerBlock; j++) {
                heightVec[nGrid * (nGridPerBlock * xBlock + i) + (nGridPerBlock * yBlock + j)] = height;
              }
            }
          }
        }
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);

      case GroundType::STEPS_INCLINE:
        heightVec.resize(nGrid * nGrid);
        for(int xBlock = 0; xBlock < nGridPerBlock; xBlock++) {
          for(int yBlock = 0; yBlock < nGridPerBlock; yBlock++) {
            double height = stepHeight * uniDist(gen);
            for(int i=0; i<nBlock; i++) {
              for(int j=0; j<nBlock; j++) {
                heightVec[nGrid * (nBlock * xBlock + i) + (nBlock * yBlock + j)] = height + xBlock * roughness;
              }
            }
          }
        }
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);

      case GroundType::STAIRS: {
        if (uniDist(gen) < (1. / 3.)) {
          nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
          stepWidth = (nGridPerBlock - 1) * gridSize;
          nGrid = static_cast<int>(size / gridSize) + 1;
          nBlock = static_cast<int>(nGrid / nGridPerBlock);
          nGrid = nBlock * nGridPerBlock;
          size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation

          heightVec.resize(nGrid * nGrid);
          double randomizedStepHeight, accumulateHeight = 0.;
          for (int xBlock = 0; xBlock < nBlock; xBlock++) {
  //          randomizedStepHeight = stepHeight * (0.75 + 0.25 * uniDist(gen));
            randomizedStepHeight = stepHeight;
            for (int i = 0; i < nGrid * nGridPerBlock; i++) {
              heightVec[xBlock * nGrid * nGridPerBlock + i] = accumulateHeight;
            }
            accumulateHeight += randomizedStepHeight;
          }
        }
        else {
          pyramidType_ = true;
          nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
          stepWidth = (nGridPerBlock - 1) * gridSize;
          nGrid = static_cast<int>(size / gridSize) + 1;
          if (nGrid % 2 == 0)
            nGrid += 1;
          size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation

          int upDown = std::round(uniDist(gen)) == 0 ? -1 : 1;
          upDown = 1;
          // Initialize the height map with zeros
          Eigen::MatrixXd heightMat;
          heightMat.setZero(nGrid, nGrid);

          double safetyZoneSize = 2.0;

          // Calculate the half dimensions of the safety zone in grid units
          int safetyZoneHalfSize = static_cast<int>((safetyZoneSize / 2) / gridSize);

          // Calculate the center of the grid
          int centerGridNum = (nGrid - 1) / 2;

          // Calculate height increment per width
          double heightIncrementPerWidth = std::tan(slopeAngle) * stepWidth;

          // Generate the pyramid
          for (int i = 0; i < nGrid; ++i) {
            for (int j = 0; j < nGrid; ++j) {
              // Check if the current cell is within the safety zone
              if (abs(i - centerGridNum) <= safetyZoneHalfSize && abs(j - centerGridNum) <= safetyZoneHalfSize) {
                heightMat(i, j) = 0; // Flat ground in the safety zone
              } else {
                // Calculate the distance from the edge of the safety zone in terms of the width steps
                int distanceFromSafetyZoneEdgeInWidthSteps = static_cast<int>((std::max(abs(i - centerGridNum), abs(j - centerGridNum)) - safetyZoneHalfSize - 1) / nGridPerBlock);
                // Calculate the height of the pyramid at this distance
                heightMat(i, j) = upDown * (distanceFromSafetyZoneEdgeInWidthSteps + 1) * heightIncrementPerWidth;
              }
            }
          }
          std::vector<double> heightVecStair(heightMat.data(), heightMat.data() + heightMat.size());
          heightVec = heightVecStair;
        }

        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);
      }

      case GroundType::NOSING_STAIRS: {
        if (uniDist(gen) < (1. / 3.)) {
          nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
          stepWidth = (nGridPerBlock - 1) * gridSize;
          nGrid = static_cast<int>(size / gridSize) + 1;
          nBlock = static_cast<int>(nGrid / nGridPerBlock);
          nGrid = nBlock * nGridPerBlock;
          size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation

          heightVec.resize(nGrid * nGrid);
          double randomizedStepHeight, accumulateHeight = 0.;
          double blockThick = std::min(stepHeight, 0.02 + 0.02 * uniDist(gen));
          double tuk = std::min(stepHeight, 0.02 + 0.02 * uniDist(gen));
          for (int xBlock = 0; xBlock < nBlock + 1; xBlock++) {
  //          randomizedStepHeight = stepHeight * (0.75 + 0.25 * uniDist(gen));
            randomizedStepHeight = stepHeight;
            auto Block = world->addBox(size, nGridPerBlock * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
            Block->setPosition(0, -size / 2 + (xBlock + 0.5) * nGridPerBlock * gridSize - tuk, -blockThick / 2 + accumulateHeight + 0.0001);
            Block->setBodyType(raisim::BodyType::STATIC);
            Block->setAppearance("blue");

            if (xBlock < nBlock) {
              for (int i = 0; i < nGrid * nGridPerBlock; i++) {
                heightVec[xBlock * nGrid * nGridPerBlock + i] = accumulateHeight;
              }
            }
            accumulateHeight += randomizedStepHeight;
          }
        }
        else {
          pyramidType_ = true;
          nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
          stepWidth = (nGridPerBlock - 1) * gridSize;
          nGrid = static_cast<int>(size / gridSize) + 1;
          if (nGrid % 2 == 0)
            nGrid += 1;
          size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation

          int upDown = std::round(uniDist(gen)) == 0 ? -1 : 1;
          upDown = 1;
          // Initialize the height map with zeros
          Eigen::MatrixXd heightMat;
          heightMat.setZero(nGrid, nGrid);

          double safetyZoneSize = 2.0;

          // Calculate the half dimensions of the safety zone in grid units
          int safetyZoneHalfSize = static_cast<int>((safetyZoneSize / 2) / gridSize);

          // Calculate the center of the grid
          int centerGridNum = (nGrid - 1) / 2;

          // Calculate height increment per width
          double heightIncrementPerWidth = std::tan(slopeAngle) * stepWidth;
          int stairNum = static_cast<int>((centerGridNum - safetyZoneHalfSize - 1) / nGridPerBlock);

          // Generate the pyramid
          for (int i = 0; i < nGrid; ++i) {
            for (int j = 0; j < nGrid; ++j) {
              // Check if the current cell is within the safety zone
              if (abs(i - centerGridNum) <= safetyZoneHalfSize && abs(j - centerGridNum) <= safetyZoneHalfSize) {
                heightMat(i, j) = 0; // Flat ground in the safety zone
              } else {
                // Calculate the distance from the edge of the safety zone in terms of the width steps
                int distanceFromSafetyZoneEdgeInWidthSteps = static_cast<int>((std::max(abs(i - centerGridNum), abs(j - centerGridNum)) - safetyZoneHalfSize - 1) / nGridPerBlock);
                // Calculate the height of the pyramid at this distance
                heightMat(i, j) = upDown * (distanceFromSafetyZoneEdgeInWidthSteps + 1) * heightIncrementPerWidth;
              }
            }
          }
          std::vector<double> heightVecStair(heightMat.data(), heightMat.data() + heightMat.size());
          heightVec = heightVecStair;

          double height = heightMat(0, centerGridNum);
          double blockThick = std::min(stepHeight, 0.02 + 0.02 * uniDist(gen));
          double tuk = std::min(stepHeight, 0.02 + 0.02 * uniDist(gen));
          int widthNum = 0, idx = 0, stairIdx = 0;
          double dist = 0., length = 0.;
          for (int i = 1; i < centerGridNum; i++) {
            if (heightMat(i, centerGridNum) != height) {
              widthNum = i - 1 - idx;
              length = safetyZoneHalfSize * gridSize * 2 + (stairNum - stairIdx) * 2 * (stepWidth + gridSize) - upDown * 2 * tuk;
              auto Block = world->addBox(length, widthNum * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
              dist = -size / 2 + idx * gridSize + widthNum * gridSize / 2.0 + upDown * tuk;
              Block->setPosition(0, dist, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
              Block->setBodyType(raisim::BodyType::STATIC);
              Block->setAppearance("blue");

              Block = world->addBox(length, widthNum * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
              dist = size / 2 - idx * gridSize - widthNum * gridSize / 2.0 - upDown * tuk;
              Block->setPosition(0, dist, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
              Block->setBodyType(raisim::BodyType::STATIC);
              Block->setAppearance("blue");

              Block = world->addBox(widthNum * gridSize, length, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
              dist = -size / 2 + idx * gridSize + widthNum * gridSize / 2.0 + upDown * tuk;
              Block->setPosition(dist, 0, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
              Block->setBodyType(raisim::BodyType::STATIC);
              Block->setAppearance("blue");

              Block = world->addBox(widthNum * gridSize, length, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
              dist = size / 2 - idx * gridSize - widthNum * gridSize / 2.0 - upDown * tuk;
              Block->setPosition(dist, 0, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
              Block->setBodyType(raisim::BodyType::STATIC);
              Block->setAppearance("blue");
              idx = i;
              stairIdx++;
            }
            height = heightMat(i, centerGridNum);
          }
        }

        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec, "default", CollisionGroup(2), CollisionGroup(1));
      }

      case GroundType::SLOPE: {
        terrainProperties_.fractalLacunarity = 1.0 + 4.0 * uniDist(gen);
        genPtr = std::make_unique<raisim::TerrainGenerator>(terrainProperties_);
        heightVec = genPtr->generatePerlinFractalTerrain();
        Eigen::MatrixXd heightMat1 = Eigen::Map<Eigen::MatrixXd>(heightVec.data(), nGrid, nGrid);

        Eigen::MatrixXd heightMat2;
        heightMat2.setZero(nGrid, nGrid);
        double theta = slopeAngle, phi = 2 * M_PI * uniDist(gen);
        phi = 0.;
        normalVector_ << std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta);
        for (int i = 0; i < nGrid; i++) {
          for (int j = 0; j < nGrid; j++) {
            heightMat2(i, j) = -(normalVector_(0) * (i - nGrid / 2) + normalVector_(1) * (j - nGrid / 2)) * gridSize / normalVector_(2);
          }
        }
        heightMat2 += heightMat1;

        std::vector<double> heightVecSlope(heightMat2.data(), heightMat2.data() + heightMat2.size());

        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVecSlope);
      }

      case GroundType::STAIRS3: {
//        nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
//        stepWidth = (nGridPerBlock - 1) * gridSize;
//        nGrid = static_cast<int>(size / gridSize) + 1;
//        nBlock = static_cast<int>(nGrid / nGridPerBlock);
//        nGrid = nBlock * nGridPerBlock;
//        size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation
//
//        heightVec.resize(nGrid * nGrid);
//        double randomizedStepHeight, accumulateHeight = 0.;
//        double blockThick = std::min(stepHeight, 0.02 + 0.02 * uniDist(gen));
//        for (int xBlock = 0; xBlock < nBlock + 1; xBlock++) {
////          randomizedStepHeight = stepHeight * (0.75 + 0.25 * uniDist(gen));
//          randomizedStepHeight = stepHeight;
//          auto Block = world->addBox(size, nGridPerBlock * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
//          Block->setPosition(0, -size / 2 + (xBlock + 0.5) * nGridPerBlock * gridSize - 0.0001, -blockThick / 2 + accumulateHeight + 0.0001);
//          Block->setBodyType(raisim::BodyType::STATIC);
//          Block->setAppearance("blue");
//
//          if (xBlock < nBlock) {
//            for (int i = 0; i < nGrid * nGridPerBlock; i++) {
//              heightVec[xBlock * nGrid * nGridPerBlock + i] = accumulateHeight;
//            }
//          }
//          accumulateHeight += randomizedStepHeight;
//        }

        pyramidType_ = true;
        nGridPerBlock = static_cast<int>(stepWidth / gridSize) + 1;
        stepWidth = (nGridPerBlock - 1) * gridSize;
        nGrid = static_cast<int>(size / gridSize) + 1;
        if (nGrid % 2 == 0)
          nGrid += 1;
        size = (nGrid - 1) * gridSize;  // remove blank part due to (int) operation

        int upDown = std::round(uniDist(gen)) == 0 ? -1 : 1;
        upDown = 1;
        // Initialize the height map with zeros
        Eigen::MatrixXd heightMat;
        heightMat.setZero(nGrid, nGrid);

        double safetyZoneSize = 2.0;

        auto Block = world->addBox(safetyZoneSize, safetyZoneSize, 0.1, 1000, "default", CollisionGroup(3), CollisionGroup(1));
        Block->setPosition(0, 0, -0.05 + 0.0001);
        Block->setBodyType(raisim::BodyType::STATIC);
        Block->setAppearance("blue");

        // Calculate the half dimensions of the safety zone in grid units
        int safetyZoneHalfSize = static_cast<int>((safetyZoneSize / 2) / gridSize);

        // Calculate the center of the grid
        int centerGridNum = (nGrid - 1) / 2;

        // Calculate height increment per width
        double heightIncrementPerWidth = std::tan(slopeAngle) * stepWidth;
        int stairNum = static_cast<int>((centerGridNum - safetyZoneHalfSize - 1) / nGridPerBlock);

        // Generate the pyramid
        for (int i = 0; i < nGrid; ++i) {
          for (int j = 0; j < nGrid; ++j) {
            // Check if the current cell is within the safety zone
            if (abs(i - centerGridNum) <= safetyZoneHalfSize && abs(j - centerGridNum) <= safetyZoneHalfSize) {
              heightMat(i, j) = 0; // Flat ground in the safety zone
            } else {
              // Calculate the distance from the edge of the safety zone in terms of the width steps
              int distanceFromSafetyZoneEdgeInWidthSteps = static_cast<int>((std::max(abs(i - centerGridNum), abs(j - centerGridNum)) - safetyZoneHalfSize - 1) / nGridPerBlock);
              // Calculate the height of the pyramid at this distance
              heightMat(i, j) = upDown * (distanceFromSafetyZoneEdgeInWidthSteps + 1) * heightIncrementPerWidth;
            }
          }
        }
        std::vector<double> heightVecStair(heightMat.data(), heightMat.data() + heightMat.size());
        heightVec = heightVecStair;

        double height = heightMat(0, centerGridNum);
        double blockThick = std::min(stepHeight, 0.01 + 0.02 * uniDist(gen));
        double tuk = 0;
        int widthNum = 0, idx = 0, stairIdx = 0;
        double dist = 0., length = 0.;
        for (int i = 1; i < centerGridNum; i++) {
          if (heightMat(i, centerGridNum) != height) {
            widthNum = i - 1 - idx;
            length = safetyZoneHalfSize * gridSize * 2 + (stairNum - stairIdx) * 2 * (stepWidth + gridSize) - upDown * 2 * tuk;
            auto Block = world->addBox(length, widthNum * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
            dist = -size / 2 + idx * gridSize + widthNum * gridSize / 2.0 + upDown * tuk;
            Block->setPosition(0, dist, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
            Block->setBodyType(raisim::BodyType::STATIC);
            Block->setAppearance("blue");

            Block = world->addBox(length, widthNum * gridSize, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
            dist = size / 2 - idx * gridSize - widthNum * gridSize / 2.0 - upDown * tuk;
            Block->setPosition(0, dist, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
            Block->setBodyType(raisim::BodyType::STATIC);
            Block->setAppearance("blue");

            Block = world->addBox(widthNum * gridSize, length, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
            dist = -size / 2 + idx * gridSize + widthNum * gridSize / 2.0 + upDown * tuk;
            Block->setPosition(dist, 0, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
            Block->setBodyType(raisim::BodyType::STATIC);
            Block->setAppearance("blue");

            Block = world->addBox(widthNum * gridSize, length, blockThick, 1000, "default", CollisionGroup(3), CollisionGroup(1));
            dist = size / 2 - idx * gridSize - widthNum * gridSize / 2.0 - upDown * tuk;
            Block->setPosition(dist, 0, -blockThick / 2 + heightMat(i - 1, centerGridNum) + 0.0001);
            Block->setBodyType(raisim::BodyType::STATIC);
            Block->setAppearance("blue");
            idx = i;
            stairIdx++;
          }
          height = heightMat(i, centerGridNum);
        }

        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec, "default", CollisionGroup(2), CollisionGroup(2));
      }

      case GroundType::TUK: {
        nGrid = (int) (size / gridSize);
        heightVec.resize(nGrid * nGrid, 0.);
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 8; k++) {
              double width = 0.03 + uniDist(gen) * 0.03;
              double height = width / 2 + 0.03 + stepHeight * (0.75 + 0.25 * uniDist(gen));
              auto box = i == 0 ? world->addBox(size, width, width, 1000) : world->addBox(width, size, width, 1000);
              int leftRight = j == 0 ? 1 : -1;
              if (i == 0) {
                box->setPosition(0, leftRight * (k + (0.9 + 0.2 * uniDist(gen))), height);
              }
              else {
                box->setPosition(leftRight * (k + (0.9 + 0.2 * uniDist(gen))), 0, height);
              }
              box->setBodyType(raisim::BodyType::STATIC);
              box->setAppearance("blue");
            }
          }
        }
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);
      }

      case GroundType::FLAT:
        nGrid = (int) (size / gridSize);
        heightVec.resize(nGrid * nGrid, 0.);
        return world->addHeightMap(nGrid, nGrid, size, size, 0., 0., heightVec);
    }
    return nullptr;
  }

 public:
  raisim::TerrainProperties terrainProperties_;
  int terrain_seed_, nGrid, nBlock, nGridPerBlock, nGridSafeZone;
  double gridSize = 0.2, size = 35.0, stepWidth, stepHeight, safeZoneSize, roughness, slopeAngle;
  Eigen::Vector3d normalVector_;
  bool pyramidType_ = false;

};

}

#endif //_RAISIM_GYM_ANYMAL_RAISIMGYM_ENV_ANYMAL_ENV_RANDOMHEIGHTMAPGENERATOR_HPP_