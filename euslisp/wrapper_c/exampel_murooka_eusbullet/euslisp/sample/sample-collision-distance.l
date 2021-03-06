(load "../eusbullet.l")
(load "irteus/demo/sample-arm-model.l")


(defun sample-collision-distance-body
    (&key
     (obj1 (make-cube 200 300 500))
     (obj2 (make-cube 400 600 1000))
     (collision-distance-func #'bt-collision-distance) ;; you can try #'pqp-collision-distance
     (obj1-coords-func
      #'(lambda (cnt) (make-coords :pos (float-vector 1000 0 0))))
     (obj2-coords-func
      #'(lambda (cnt)
          (make-coords
           :pos (float-vector (* 1500 (sin (/ cnt 100.0))) (* 500 (sin (+ (/ cnt 200.0) (deg2rad 45)))) 0)
           :rpy (list (* pi (sin (/ cnt 200.0))) (+ (* pi (sin (/ cnt 400.0))) pi/2) 0)
           )))
     )
  (let* ((cnt 0)
         (ret)
         )
    (when obj1-coords-func
      (send obj1 :newcoords (funcall obj1-coords-func (float cnt))))
    (when obj2-coords-func
      (send obj2 :newcoords (funcall obj2-coords-func (float cnt))))
    (send obj1 :set-color (float-vector 1 0 0) 0.5)
    (send obj2 :set-color (float-vector 0 1 0) 0.4)
    (objects (list obj1 obj2))

    (do-until-key
     ;; move object
     (incf cnt)
     (when obj1-coords-func
       (send obj1 :newcoords (funcall obj1-coords-func (float cnt))))
     (when obj2-coords-func
       (send obj2 :newcoords (funcall obj2-coords-func (float cnt))))
     (send *irtviewer* :draw-objects)
     ;; get distance beween target-link and obj
     (setq ret (funcall collision-distance-func obj1 obj2))
     ;; draw
     (send (elt ret 1) :draw-on :flush nil :width 16 :size 50 :color (float-vector 1 0.4 0.4))
     (send (elt ret 2) :draw-on :flush nil :width 16 :size 50 :color (float-vector 0.4 1 0.4))
     (send (make-line (elt ret 1) (elt ret 2)) :draw-on :flush nil
           :width 8 :color (if (> (elt ret 0) 0) (float-vector 0 1 1) (float-vector 1 1 0)))
     (send *irtviewer* :viewer :flush)
     (unix::usleep (* 20 1000))
     (x::window-main-one)
     )
    ))
(warn "(sample-collision-distance-body)~%")

(defun sample-collision-distance-sphere
    ()
  (sample-collision-distance-body
   :obj2 (make-sphere 600))
  )

(defun sample-collision-distance-cube
    ()
  (sample-collision-distance-body)
  )

(defun sample-collision-distance-cylinder
    ()
  (sample-collision-distance-body
   :obj2 (make-cylinder 400 1200)
   )
  )

(defun sample-collision-distance-conv
    ()
  (sample-collision-distance-body
   :obj2 (make-cone (float-vector 0 0 1500) (list (float-vector -800 -500 0) (float-vector 800 -500 0) (float-vector 0 500 0)))
   )
  )


(defun sample-collision-distance-link
    (&key
     (obj (make-cube 200 500 500))
     (collision-distance-func #'bt-collision-distance)
     (obj-coords-func
      #'(lambda (cnt) (make-coords :pos (float-vector 500 0 250))))
     )
  (let* ((cnt 0)
         (ret)
         (robot (instance sarmclass :init))
         (target-link (elt (send robot :links) 4))
         (base-link (elt (send robot :links) 0))
         )
    (when obj-coords-func
      (send obj :newcoords (funcall obj-coords-func (float cnt))))
    (send obj :set-color (float-vector 1 0 0) 0.5)
    (objects (list robot obj))

    (do-until-key
     ;; move object and robot
     (incf cnt)
     (dolist (j (send robot :joint-list))
       (send j :joint-angle
             (+ (* 0.49 (- (send j :max-angle) (send j :min-angle)) (sin (/ cnt 100.0)))
                (* 0.5 (+ (send j :max-angle) (send j :min-angle)))))
       )
     (when obj-coords-func
       (send obj :newcoords (funcall obj-coords-func (float cnt))))
     (send obj :newcoords (funcall obj-coords-func (float cnt)))
     (send *irtviewer* :draw-objects)
     ;; get distance beween target-link and obj
     (setq ret (funcall collision-distance-func target-link obj))
     (send (elt ret 1) :draw-on :flush nil :width 16 :size 50 :color (float-vector 1 0.4 0.4))
     (send (elt ret 2) :draw-on :flush nil :width 16 :size 50 :color (float-vector 0.4 1 0.4))
     (send (make-line (elt ret 1) (elt ret 2)) :draw-on :flush nil
           :width 8 :color (if (> (elt ret 0) 0) (float-vector 0 1 1) (float-vector 1 1 0)))
     ;; get distance beween target-link and base-link
     (setq ret (funcall collision-distance-func target-link base-link))
     ;; draw
     (send (elt ret 1) :draw-on :flush nil :width 16 :size 50 :color (float-vector 1 0.4 0.4))
     (send (elt ret 2) :draw-on :flush nil :width 16 :size 50 :color (float-vector 0.4 1 0.4))
     (send (make-line (elt ret 1) (elt ret 2)) :draw-on :flush nil
           :width 8 :color (if (> (elt ret 0) 0) (float-vector 0 1 1) (float-vector 1 1 0)))
     (send *irtviewer* :viewer :flush)
     (unix::usleep (* 20 1000))
     (x::window-main-one)
     )
    ))
(warn "(sample-collision-distance-link)~%")


(defun sample-collision-distance-2d-analytical
    (&key
     (max-cnt)
     (visualize? t)
     )
  (let* ((radius1 100)
         (radius2 200)
         (obj1 (make-sphere radius1))
         (obj2 (make-sphere radius2))
         (cnt 0)
         (bt-dist)
         (analy-dist)
         (ret)
         )
    (send obj1 :set-color (float-vector 1 0 0) 0.5)
    (send obj2 :set-color (float-vector 0 1 0) 0.4)
    (when visualize?
      (objects (list obj1 obj2)))

    (do-until-key
      ;; move object
      (incf cnt)
      (send obj1 :newcoords (make-coords :pos (float-vector (* 500.0 (sin (/ cnt 100.0))) 50 0)))
      (when visualize?
        (send *irtviewer* :draw-objects))
      ;; get bullet distance
      (setq ret (bt-collision-distance obj1 obj2))
      (setq bt-dist (elt ret 0))
      ;; get analytical distance and compare
      (setq analy-dist (- (norm (send obj1 :worldpos)) (+ radius1 radius2)))
      (assert (eps= bt-dist analy-dist 1e-3))
      ;; draw
      (when visualize?
        (send (elt ret 1) :draw-on :flush nil :width 16 :size 50 :color (float-vector 1 0.4 0.4))
        (send (elt ret 2) :draw-on :flush nil :width 16 :size 50 :color (float-vector 0.4 1 0.4))
        (send (make-line (elt ret 1) (elt ret 2)) :draw-on :flush nil
              :width 8 :color (if (> (elt ret 0) 0) (float-vector 0 1 1) (float-vector 1 1 0)))
        (send *irtviewer* :viewer :flush)
        (unix::usleep (* 20 1000))
        (x::window-main-one)
        )
      (when (and max-cnt (> cnt max-cnt))
        (return-from nil nil))
      )
    ))
(warn "(sample-collision-distance-2d-analytical)~%")
