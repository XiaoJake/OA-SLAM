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


#include "ImageDetections.h"

#include <filesystem>
#include <unordered_set>


namespace fs = std::filesystem;


namespace ORB_SLAM2
{

std::ostream& operator <<(std::ostream& os, const Detection& det)
{
    os << "Detection:  cat = " << det.category_id << "  score = "
       << det.score << "  bbox = " << det.bbox.transpose();
    return os;
}


ImageDetectionsManager::ImageDetectionsManager(const std::string& filename, const std::vector<int>& cats_to_ignore)
{
    std::unordered_set<unsigned int> to_ignore(cats_to_ignore.begin(), cats_to_ignore.end());
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Warning failed to open file: " << filename << std::endl;
        return ;
    }
    fin >> data_;

    for (auto& frame : data_)
    {
        std::string name = frame["file_name"].get<std::string>();
        name = fs::path(name).filename();
        frame_names_.push_back(name);

        std::vector<Detection::Ptr> detections;
        for (auto& d : frame["detections"])
        {
            double score = d["detection_score"].get<double>();
            unsigned int cat = d["category_id"].get<unsigned int>();
            if (to_ignore.find(cat) != to_ignore.end())
                continue;
            auto bb = d["bbox"];
            Eigen::Vector4d bbox(bb[0], bb[1], bb[2], bb[3]);
            detections.push_back(std::shared_ptr<Detection>(new Detection(cat, score, bbox)));
        }
        detections_[name] = detections;
    }
}


std::vector<Detection::Ptr> ImageDetectionsManager::get_detections(const std::string& name) const {
    std::string basename = fs::path(name).filename();

    if (detections_.find(basename) == detections_.end())
        return {};
    return detections_.at(basename);
}
std::vector<Detection::Ptr> ImageDetectionsManager::get_detections(unsigned int idx) const {
    if (idx < 0 || idx >= frame_names_.size()) {
        std::cerr << "Warning invalid index: " << idx << std::endl;
        return {};
    }
    return this->get_detections(frame_names_[idx]);
}

}