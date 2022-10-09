/**
* This file is part of OA-SLAM.
*
* Copyright (C) 2022 Matthieu Zins <matthieu.zins@inria.fr>
* (Inria, LORIA, Université de Lorraine)
* OA-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OA-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OA-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef IMAGE_DETECTIONS_H
#define IMAGE_DETECTIONS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include "Utils.h"


namespace ORB_SLAM2
{

class Detection
{
public:
    typedef std::shared_ptr<Detection> Ptr;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Detection(unsigned int cat, double det_score, const BBox2 bb)
    : category_id(cat), score(det_score), bbox(bb) {}


    friend std::ostream& operator <<(std::ostream& os, const Detection& det);
    unsigned int category_id;
    double score;
    BBox2 bbox;

private:
    Detection() = delete;


};

// using vector_Detection = std::vector<Detection, Eigen::aligned_allocator<Detection>>;

class ImageDetectionsManager
{
public:
    ImageDetectionsManager(const std::string& filename, const std::vector<int>& cats_to_ignore={});


    std::vector<Detection::Ptr> get_detections(const std::string& name) const;
    std::vector<Detection::Ptr> get_detections(unsigned int idx) const;

private:
    ImageDetectionsManager() = delete;
    std::unordered_map<std::string, std::vector<Detection::Ptr>> detections_;
    std::vector<std::string> frame_names_;
    json data_;
};

}

#endif