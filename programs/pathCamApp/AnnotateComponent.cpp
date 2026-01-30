//
//  AnnotateComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

void AnnotateComponent::removeSelected()
{
  if (!selected)
    return;

  // Remove annotation object from the annotations list
  auto itAnno = std::find(annotations->begin(), annotations->end(), selected);
  if (itAnno == annotations->end())
    return;

  if (selected->getType() == Annotation::_SEG)
  {
    // Safe cast (never dereference a failed dynamic_cast)
    auto seg = std::dynamic_pointer_cast<SegmentAnnotation>(selected);
    if (seg)
    {
      const int targetSegID = seg->ID;

      auto* mr0 = rightComponent->MRImageSet->MRImages[0].get();
      for (auto& tileIdx : mr0->liveTiles){
        auto tileObj = mr0->get_base_tile(tileIdx);

        auto& masks = tileObj->SAMMasks; // std::map<int, std::pair<cv::cuda::GpuMat, void*>>

        auto itMask = masks.find(targetSegID);
        if (itMask != masks.end()){
          // Release GPU memory
          itMask->second.first.release();

          // Delete frontend cached object, then null it
          if (itMask->second.second)
          {
            delete static_cast<juce::Image*>(itMask->second.second);
            itMask->second.second = nullptr;
          }

          // Remove the entry from the map
          masks.erase(itMask);
        }
      }
    }
  }

  annotations->erase(itAnno);
  selected.reset();

  leftComponent->updatelist();
  repaint();
}

void AnnotateComponent::setImage(std::shared_ptr<MRTiledImageSet> image)  {
  parent->MRimage->annotations = eraseAnnotations(annotations);
  rightComponent->setImage(image);
  if (image && image->annotations && !image->annotations->empty()) {
    annotations = restoreAnnotations(image->annotations);
  }
  else {
    annotations->clear();
  }
}

