//
//  AnnoViewComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"

AnnoViewComponent::AnnoViewComponent(AnnotateComponent *parent, std::shared_ptr<fRectangle> view,
                                     juce::StringArray &iconNames, OwnedArray<juce::Drawable> &iconsFromZipFile,
                                     std::shared_ptr<std::vector<std::shared_ptr<Annotation> > >
                                     annotations) : ImageViewComponent(view, iconNames, iconsFromZipFile,
                                                                       parent->parent), annotateParent(parent),
                                                    annotations(annotations) {
  annotateOverlay.reset(new AnnotateOverlay(this, iconNames, iconsFromZipFile));
  addAndMakeVisible(annotateOverlay.get());
}

void AnnoViewComponent::toggleMode(int mode) { annotateParent->toggleMode(mode); }
void AnnoViewComponent::setMode(int mode) { annotateParent->setMode(mode); }

int AnnoViewComponent::getMode() { return annotateParent->getMode(); }
void AnnoViewComponent::setSelected(std::shared_ptr<Annotation> annotation) { annotateParent->setSelected(annotation); }


bool AnnoViewComponent::keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) {
  ImageViewComponent::keyPressed(key, originatingComponent);

  if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_SEG) {
    if (key == juce::KeyPress::createFromDescription("q")) {
      std::cout << "Run SAM\n";
      SegmentAnnotation *cast = dynamic_cast<SegmentAnnotation *>(annotateParent->getSelected().get());
      //cast->ID = parent->segID++;
      auto input = cast->input;
      input.push_back(Point3f(cast->fovUpperLeftCorner.x, cast->fovUpperLeftCorner.y, 4));
      input.push_back(Point3f(cast->fovLowerRightCorner.x, cast->fovLowerRightCorner.y, 5));
      annotateParent->parent->sCam->segment_with_SAM(input, cast->ID, MRImageSet->index);
      repaint();
      return true; // Key press handled
    }
  }


  if (key == juce::KeyPress::createFromDescription("shift + d")) {
    std::shared_ptr<Annotation> new_annotation;
    new_annotation.reset(new debugAnnotation("0 0", MRImageSet));
    new_annotation->setName("0 0");
    annotateParent->setSelected(new_annotation);
    annotations->push_back(new_annotation);
    annotateParent->annotationsUpdated();
    debugIterAnnotation = new_annotation;
    debugIterCount = 0;
    debugIterMode = true;
    // repaint();
    parent->startCompositingUIUpdates();
    parent->notify_new_data();
    return true;
  }

  return false; // Key press not handled
}

bool AnnoViewComponent::polyMouseDown(const juce::MouseEvent &event) {
  // Right click without shift: open dropdown menu for preconfigured class selection
  if (event.mods.isRightButtonDown() && !event.mods.isShiftDown()) {
    const auto clickView = screen2view(fPoint(event.x, event.y), *view);

    std::shared_ptr<PolygonAnnotation> hitPoly;
    for (auto it = annotations->rbegin(); it != annotations->rend(); ++it) {
      auto poly = std::dynamic_pointer_cast<PolygonAnnotation>(*it);
      if (poly && poly->containsPoint(clickView)) {
        hitPoly = poly;
        break;
      }
    }
    if (!hitPoly) return false;

    juce::PopupMenu m;
    for (int i = 0; i < annotateParent->parent->sCam->preconfiguredAnnoLabels.size(); ++i) {
      m.addItem(i + 1, annotateParent->parent->sCam->preconfiguredAnnoLabels[i].name);
    }
    const auto screenPt = event.getScreenPosition();
    juce::Rectangle<int> anchor(screenPt.x, screenPt.y, 1, 1);

    m.showMenuAsync(
      juce::PopupMenu::Options().withTargetScreenArea(anchor),
      [this, hitPoly](int result) {
        if (result <= 0) return;
        if (result >= 1 && result <= (int) annotateParent->parent->sCam->preconfiguredAnnoLabels.size()) {
          auto ci = annotateParent->parent->sCam->preconfiguredAnnoLabels[(size_t) (result - 1)];
          hitPoly->setName(ci.name);
          hitPoly->setColor(Colour(ci.r, ci.g, ci.b));
          annotateParent->annotationsUpdated();
          repaint();
        }
      });

    return true;
  }

  // Shift + Left click: add a point to polygon
  if (event.mods.isLeftButtonDown() && event.mods.isShiftDown()) {
    if (annotateParent->getMode() == Annotation::_POLY) {
      std::shared_ptr<Annotation> new_annotation;
      new_annotation.reset(new PointClickPoly("Polygon"));
      annotateParent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      annotateParent->annotationsUpdated();
      annotateParent->setMode(Annotation::_NONE);
    }

    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
      PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
      cast->add(screen2view(fPoint(event.x, event.y), *view));
    }

    // // Always consume shift+left click to prevent navigation interference
    // return true;
  }

  // Left click without shift: select point for dragging
  if (event.mods.isLeftButtonDown() && !event.mods.isShiftDown()) {
    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
      PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
      if (cast->test(screen2view(fPoint(event.x, event.y), *view), screen2viewScale(*view))) {
        return true;
      }
    }
  }

  return false;
}

bool AnnoViewComponent::polyMouseDrag(const juce::MouseEvent &event) {
  // If shift is held, consume the event to prevent navigation interference
  if (event.mods.isShiftDown()) {
    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
      PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
      if (cast->isPointSelected()) {
        cast->move(screen2view(fPoint(event.x, event.y), *view));
      }
    }
    return true;
  }

  // Non-shift dragging for point selection
  if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
    PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
    if (cast->isPointSelected()) {
      cast->move(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }
  }
  return false;
}

bool AnnoViewComponent::polyMouseUp(const juce::MouseEvent &event) {
  // If shift is held, consume the event to prevent navigation interference
  if (event.mods.isShiftDown()) {
    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
      PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
      cast->unSelect();
    }
    return true;
  }

  // Non-shift mouse up for point deselection
  if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_POLY) {
    PointClickPoly *cast = dynamic_cast<PointClickPoly *>(annotateParent->getSelected().get());
    cast->unSelect();
    return true;
  }

  return false;
}

bool AnnoViewComponent::measureMouseDown(const juce::MouseEvent &event) {
  if (event.mods.isRightButtonDown()) {
    if (annotateParent->getMode() == Annotation::_MEAS) {
      std::shared_ptr<Annotation> new_annotation;
      new_annotation.reset(new MeasureAnnotation("Measure"));
      annotateParent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      annotateParent->annotationsUpdated();
      annotateParent->setMode(Annotation::_NONE);
    }

    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_MEAS) {
      MeasureAnnotation *cast = dynamic_cast<MeasureAnnotation *>(annotateParent->getSelected().get());
      if (cast->isMeasuring()) {
        cast->stopMeasuring(screen2view(fPoint(event.x, event.y), *view));
      } else {
        cast->startMeasuring(screen2view(fPoint(event.x, event.y), *view));
      }
      return true;
    }
  }
  return false;
}

bool AnnoViewComponent::measureMouseMove(const juce::MouseEvent &event) {
  if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_MEAS) {
    MeasureAnnotation *cast = dynamic_cast<MeasureAnnotation *>(annotateParent->getSelected().get());
    if (cast->isMeasuring()) {
      cast->continueMeasuring(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }
  }

  return false;
}

bool AnnoViewComponent::segMouseDown(const juce::MouseEvent &event) {
  if (event.mods.isLeftButtonDown() && event.mods.isShiftDown()) {
    if (annotateParent->getMode() == Annotation::_SEG) {
      auto seg = new SegmentAnnotation("SAM");
      seg->ID = annotations->size();
      std::shared_ptr<Annotation> new_annotation;
      new_annotation.reset(seg);
      annotateParent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      annotateParent->annotationsUpdated();
      annotateParent->setMode(Annotation::_NONE);
    }

    if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_SEG) {
      SegmentAnnotation *cast = dynamic_cast<SegmentAnnotation *>(annotateParent->getSelected().get());

      fPoint fovUpperLeft = view->getTopLeft();
      fPoint fovLowerRight = view->getBottomRight();
      cast->update_FOV(fovUpperLeft, fovLowerRight);

      fPoint temp = screen2view(fPoint(event.x, event.y), *view);
      Point3f point(temp.x, temp.y, 1.0);
      cast->add(point);
      //parent->parent->sCam->as->on_click(point);
    }

    // Always consume shift+left click to prevent navigation interference
    return true;
  }

  // if (event.mods.isRightButtonDown() && event.mods.isShiftDown()) {
  //   if (annotateParent->getSelected() != NULL && annotateParent->getSelected()->getType() == Annotation::_SEG) {
  //     SegmentAnnotation *cast = dynamic_cast<SegmentAnnotation *>(annotateParent->getSelected().get());
  //
  //     fPoint fovUpperLeft = screen2view(view->getTopLeft(), *view);
  //     fPoint fovLowerRight = screen2view(view->getBottomRight(), *view);
  //     cast->update_FOV(fovUpperLeft, fovLowerRight);
  //
  //     fPoint temp = screen2view(fPoint(event.x, event.y), *view);
  //     Point3f point(temp.x, temp.y, 0.0);
  //     cast->add(point);
  //     //parent->parent->sCam->as->on_click(point);
  //   }
  //
  //   // Always consume shift+right click to prevent navigation interference
  //   return true;
  // }

  return false;
}

bool AnnoViewComponent::segMouseUp(const juce::MouseEvent &event) {
  if ((annotateParent->getMode() == Annotation::_SEG || annotateParent->getSelected()->getType() == Annotation::_SEG) &&
      event.mods.
      isShiftDown()) {
    return true;
  }
  return false;
}

bool AnnoViewComponent::segMouseDrag(const juce::MouseEvent &event) {
  if ((annotateParent->getMode() == Annotation::_SEG || annotateParent->getSelected()->getType() == Annotation::_SEG) &&
      event.mods.
      isShiftDown()) {
    return true;
  }
  return false;
}

void AnnoViewComponent::mouseDown(const juce::MouseEvent &event) {
  bool handled = false;
  {
    const ScopedLock lock(mutex);
    handled = polyMouseDown(event);
    if (!handled) { handled = measureMouseDown(event); }
    if (!handled) { handled = segMouseDown(event); }
    if (handled) { repaint(); }
  }

  if (!handled) {
    ImageViewComponent::mouseDown(event);
  }
}

void AnnoViewComponent::mouseUp(const juce::MouseEvent &event) {
  bool handled;
  {
    const ScopedLock lock(mutex);
    handled = polyMouseUp(event);
    if (!handled) { handled = segMouseUp(event); }
    if (handled) { repaint(); }
  }

  if (!handled) {
    ImageViewComponent::mouseUp(event);
  }
}

void AnnoViewComponent::mouseDrag(const juce::MouseEvent &event) {
  bool handled;
  {
    const ScopedLock lock(mutex);
    handled = polyMouseDrag(event);
    if (!handled) { handled = segMouseDrag(event); }
    if (handled) { repaint(); }
  }

  if (!handled) {
    ImageViewComponent::mouseDrag(event);
  }
}

void AnnoViewComponent::mouseMove(const juce::MouseEvent &event) {
  bool handled;
  {
    const ScopedLock lock(mutex);
    handled = measureMouseMove(event);
    if (handled) { repaint(); }
  }

  if (!handled) {
    ImageViewComponent::mouseMove(event);
  }
}

void AnnoViewComponent::paint(juce::Graphics &g) {
  ImageViewComponent::paint(g);
  {
    const ScopedLock lock(mutex);
    for (unsigned int i = 0; i < annotations->size(); i++) {
      bool isSelected = (*annotations)[i].get() == annotateParent->getSelected().get();
      float alpha = isSelected ? annotateParent->selectedAlpha : annotateParent->unselectedAlpha;
      (*annotations)[i]->paint(g, view->getPosition(), isSelected,
                               view2screenScale(*view), alpha);
    }
  }
}
