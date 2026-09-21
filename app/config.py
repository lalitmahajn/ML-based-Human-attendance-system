"""Single source of truth for every tunable.

The previous codebase had the recognition threshold in three places with three
different values (0.85 in env.example, 0.62 in constants.py, and the one that
actually ran: 0.5 hardcoded).  Everything here is read from the environment or
a .env file exactly once, at import.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- paths ----------------------------------------------------------
    root: Path = ROOT
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    media_dir: Path = ROOT / "media"
    gallery_dir: Path = ROOT / "face_id_users"

    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'ematsy.db'}"
    database_url_sync: str = f"sqlite:///{ROOT / 'data' / 'ematsy.db'}"

    # Live preview WebSocket streaming FPS to web frontend
    live_preview_fps: int = 30

    # ---- models ---------------------------------------------------------
    detector_kind: str = "yolo"                       # only option; YuNet removed
    # CrowdHuman YOLOv8n (person + head). Used only for tracking - see the note
    # in app/core/head_detector.py. Set to "" to fall back to face-box tracking.
    # Exported from the same yolov8n_640.pt weights at a 960 input. Training
    # size and inference size are independent for a fully-convolutional YOLO, so
    # no retraining was needed - only a re-export.
    #
    # Why 960: after letterboxing a 4K frame, a head reaches the model at 16 px
    # median at 640 input, which is right at the floor for small-object
    # detection. At 960 it arrives at 24 px. Measured on 25 real frames: 30
    # heads at 640 vs 32 at 960 (5.9 -> 12.0 ms). 1280 found no more than 960
    # while costing 21.7 ms, so the gain stops there.
    # Retrained 100 epochs on full CrowdHuman (mAP50 .812, mAP50-95 .509) and
    # exported at the frame's native 16:9 rather than a square. A 960x540 frame
    # letterboxed to 960x960 spends 44% of every buffer on grey padding that the
    # CPU converts and the GPU convolves. Measured against the old 960x960
    # export on 168 real corridor frames: finds all 111 heads it found plus 15
    # more (9-14 px, distant), mean confidence .730 vs .713, median IoU .915 -
    # strictly better, and detect() drops 60.0 -> 22.7 ms.
    # fp16. Validated on 168 real corridor frames against the fp32 export:
    # identical 126 detections, zero found by only one of them, mean confidence
    # 0.7295 vs 0.7296, median IoU 0.9848 - while GPU inference drops 4.33 ->
    # 3.46 ms and the file halves to 6.2 MB.
    head_model: str = "yolov8n_head_960x544_fp16.onnx"
    head_input: int = 960          # fallback only; geometry comes from the model
    head_conf: float = 0.35
    # Which class ByteTrack associates on. The detector finds both in one pass.
    # Measured over 6 real clips, same detections either way:
    #   head   -> 13 tracks, median  9 frames, 31% fragments
    #   person ->  7 tracks, median 188 frames, 14% fragments
    # A body is larger and more persistent, so association survives the moments
    # a head turns away or is briefly missed. Recognition still runs on the head
    # box found inside each tracked person, and the trajectory still follows a
    # head-shaped point so the calibrated direction geometry is unaffected.
    track_on: str = "person"           # person | head
    detector_model: str = "yolov8n-face.pt"
    aligner_model: str = "dfa_mobilenet_aligner.onnx"
    # Fine-tuned AdaFace IR-101 (NIST FRVT submission build). Exported with
    # a dynamic batch dim and a named "embedding" output, so no graph surgery
    # is needed. Genuinely different weights from the base: mean cosine 0.53.
    # fp16. Full-gallery revalidation is numerically indistinguishable from
    # fp32 - d-prime 10.03 both, rank-1 100%, genuine mean 0.8484 both, weakest
    # genuine pair 0.4551 -> 0.4557 - so the calibrated threshold still applies.
    # 6.19 -> 4.57 ms per batch, and 261 MB -> 130 MB.
    # Switching this is the only change needed: `threshold_for()` resolves the
    # matching threshold from the model itself. Two things do NOT follow
    # automatically and are checked rather than trusted:
    #   * the gallery must be rebuilt (`scripts/enroll.py`) - `load_gallery()`
    #     refuses a gallery built by a different recognizer;
    #   * the encrypted model must exist for a licensed deployment
    #     (`scripts/encrypt_models.py`), or the release cannot load it.
    recognizer_model: str = "adaface_ir101_finetune_fp16.onnx"

    # ---- person re-identification ---------------------------------------
    # Body ReID: it links one unrecognised person across the two cameras, and
    # it is one half of the pseudo-person grouping below. NEVER an identity for
    # attendance on its own - measured on this corpus, naming an unknown by
    # body alone mislabels 10.9% of confirmed VISITORS at the matching
    # threshold, and no threshold makes the wrong count meaningfully smaller
    # than the right one (integration/docs/HISOBOT_YUZ_TANA.md, section 5).
    #
    # E11: OSNet-x0.75, 512-d, exported from the checkpoint that won
    # phd/dissertatsiya2's enrolment-protocol evaluation. It REPLACED
    # `reid_resnet101_ibn_256x128_e2048_fp16.onnx`, which turned out to have
    # been exported from an E1-era checkpoint - 6.5 FNIR points worse than the
    # ResNet checkpoints it was supposed to carry, confirmed by MD5 and by a
    # 0.31-0.47 cosine between the ONNX and the PyTorch weights.
    #
    #                    old (R101-IBN)   new (OSNet-x0.75)
    #   file size             85 MB          2.7 MB   (31x)
    #   compute             6.52 GFLOPs      0.60     (10.9x)
    #   embedding             2048            512
    #   FNIR@10% (20260902)  18.22          11.74
    #
    # MEASURED ON THIS CORRIDOR, 2026-09-08, same 281-pass corpus and the same
    # protocol as the threshold below (bench/reid_match_eval.py):
    #
    #                    old            new
    #   genuine median   0.700          0.787
    #   impostor MAX     0.585          0.512
    #   at 0.677/0.05    15 correct     29 correct, both 0 wrong
    #
    # So the operating point did not have to move to keep the same precision -
    # recall very nearly doubled at it, and the worst impostor fell further
    # below it. See the threshold note for why it was kept anyway.
    #
    # Empty disables the whole ReID path - the pipeline then behaves exactly as
    # it did before, which is what a deployment without the model should get.
    #
    # WARNING: `reid_pass.vector` from the old model is in a different space.
    # `_match` and the pseudo gallery both filter on `model_name`, so the two
    # never mix - but passes stored before the swap cannot match passes stored
    # after it. The changeover belongs on a day boundary.
    reid_model: str = "reid_osnet_x0_75_256x128_e512_fp16.onnx"

    # How often a body crop is kept per person-pass, and the ceiling per pass.
    # 2 s over a median 9.8 s pass is ~5 crops; the cap stops somebody standing
    # in the corridor from filling the disk on their own. 0 disables collection
    # while leaving the rest of the ReID path intact.
    # How often a body crop is COLLECTED. 1 s, not 2: the median pass is under
    # ten seconds, so a 2 s cadence yielded four crops and no room to choose
    # between them. What is kept is still spread across the whole pass - see
    # body_crop_keep_per_pass - so collecting more often buys choice, not
    # clustering. A frame rejected for having no visible head does NOT advance
    # this timer, so the next usable frame is taken immediately.
    body_crop_interval_s: float = 1.0
    # COLLECTED per pass, before selection. At 1 s this covers a 30 s pass; a
    # loiterer is bounded here rather than filling the disk. You cannot spread
    # a selection over samples you never took, which is why this is generous.
    body_crop_max_per_pass: int = 30
    # KEPT per pass, chosen as far apart in time as the pass allows. Ten is
    # what a ReID tracklet needs; the first ten in arrival order would all come
    # from the first twenty seconds, at one distance and one angle. A short
    # pass simply keeps everything it collected at the 2 s cadence.
    # Crops are stored at this width. 256x128 is what the model consumes, so
    # 320 leaves headroom for re-cropping without keeping 4K regions.
    body_crop_keep_per_pass: int = 10
    body_crop_width: int = 320

    # Cosine similarity for calling two passes the same person across cameras.
    # 0.6769 is the measured FPIR=1% operating point from the research project's
    # cross-camera evaluation, reproduced by this export at 0.68
    # (bench/eval_reid.py). PROVISIONAL for this corridor: it comes from a
    # 45-person test set, and the face threshold taught the same lesson - the
    # deployed 0.18 turned out to sit BELOW the worst real impostor.
    #
    # MEASURED on this corridor 2026-09-01 (bench/reid_match_eval.py, 281 passes,
    # 128 face-labelled, gallery = Entrance 59 -> query = Exit 69, using the face
    # path's own identities as ground truth):
    #
    #   genuine  median 0.700  min 0.441
    #   impostor MAX    0.585          <- the number that matters
    #   at 0.677 + margin 0.05:  15 correct, 0 wrong of 69 queries
    #
    # So 0.6769 sits comfortably above the worst impostor seen here, unlike the
    # face threshold's first outing. Dropping to 0.55 nearly doubles recall (27
    # correct, still 0 wrong) and is tempting - but only 10 of those queries were
    # true impostors, because most Exit passes in this corpus DO have a matching
    # entrance. In production most unknowns will not, so the impostor rate will
    # be far higher than this sample suggests. Kept conservative until measured
    # against a realistic mix.
    #
    # RE-MEASURED 2026-09-08 for the OSNet model, same corpus and protocol:
    #
    #   genuine  median 0.787  min 0.486
    #   impostor MAX    0.512          <- fell from 0.585
    #   at 0.6769 + margin 0.05:  29 correct, 0 wrong of 69 queries
    #
    # The number is UNCHANGED on purpose. A threshold is not transferable
    # between models and had to be re-derived - but re-deriving it landed in
    # the same place, because the new model widened the gap on both sides
    # rather than shifting the scale. Recall at this point went 15 -> 29 for
    # free. The one thing that did change is the headroom: 0.6769 now sits 0.16
    # above the worst impostor instead of 0.09, so the same conservatism buys
    # more than it did.
    reid_match_threshold: float = 0.6769
    # The runner-up must be beaten by this much, for the reason the face
    # matcher has the same rule: the genuine and impostor distributions overlap
    # heavily here (wrong_sim_p90 0.664 against right_sim_p10 0.521), so a top
    # score alone is weak evidence.
    reid_match_margin: float = 0.05
    # A pass with fewer crops than this is not matched: a tracklet feature
    # built from one blurred crop is not worth a cross-camera claim.
    reid_min_crops: int = 2
    reid_retention_days: int = 30
    # Let the FACE break a cross-camera tie the body could not
    # (`reid_worker._match`).
    #
    # MEASURED, this corridor, 281 passes (bench/reid_match_eval.py --sweep):
    # the runner-up margin costs more than half the correct matches - 55
    # accepts fall to 29 at 0.6769 - while the wrong count is zero either way.
    # Those are not impostors being caught. They are passes where two people's
    # CLOTHING scored alike and the body axis had nothing left to say.
    #
    # So when the body is undecided, and only then, the face is asked: among
    # the candidates that already clear `reid_match_threshold` on body, one
    # must clear `pseudo_face_threshold` on face and beat the runner-up's face
    # by `reid_match_margin`. The calibrated accept decision is untouched -
    # every candidate still has to clear the body bar - so this can only ever
    # recover a match the margin threw away, never admit a new kind.
    #
    # NOT a blended score. The research fusion (global-z blend, w = 0.6,
    # FNIR@10% 11.59 -> 5.93 over three days, p < 1e-38) was measured under the
    # ENROLMENT protocol, where the decision is a threshold on the blend. This
    # is a pairwise link with a runner-up margin, and a z-normalised blend is
    # not on the scale `reid_match_threshold` was calibrated against. Tried:
    # blending changed the ranking with no calibrated bar to judge it by, and
    # since a flipped winner almost never also clears the body margin, it
    # converted accepts into rejections instead of improving them - a stricter
    # matcher wearing fusion's name.
    #
    # false disables it and restores body-only matching.
    reid_match_face_tiebreak: bool = True

    # ---- second-chance face recognition, at TRACKLET level ---------------
    # The live vote judges every FRAME on its own and then requires
    # `vote_min_recognitions` of them to agree. A pass that yields three good
    # frames therefore names nobody however clearly each of them scores - the
    # evidence exists, the rule simply cannot reach it.
    #
    # So when a pass ends unnamed, its embedded frames are combined into ONE
    # quality-weighted template and matched against the same gallery once more.
    # No new model, no new gallery, no new inference: the embeddings were
    # already computed on the capture thread and are being thrown away.
    #
    # TWO MEASUREMENTS, AND THEY ARE NOT THE SAME MEASUREMENT.
    #
    # The research figure (integration/docs/HISOBOT_YUZ_TANA.md, section 4) is
    # +23...30% more employee passes per day at 97.4% precision. It was made on
    # stored 320-px body crops with the live quality gates RELAXED, so it says
    # what a whole-pass template could recover - not what this rule recovers.
    # THIS rule combines only the frames that already PASSED the gates.
    #
    # MEASURED ON THE DEPLOYED RULE, 2026-09-08, by running the real pipeline
    # over 191 recorded clips of 2026-08-26 (bench/tracklet_recheck_eval.py,
    # --stride 4, 66 completed passes):
    #
    #   agreement with the live vote where it named somebody : 22/22 (100%)
    #   disagreements, sweeping the threshold 0.20 -> 0.40   : 0 at every point
    #   recovered from the 38 unnamed passes                 : 2  (+7%)
    #
    # So: safe, and worth having, but +7% rather than +23%, on a small sample
    # of one day. The difference is the gates - `min_aligner_score` rejects
    # frames before they are ever embedded, so the template never sees them.
    # Embedding gated frames is the next lever and it is NOT free: it adds work
    # to a capture thread with a 50 ms budget, so it needs measuring before it
    # is attempted. Do not quote the research number for this rule.
    #
    # Dropping to 0.20 raises recovery 2 -> 5 with still zero disagreements.
    # Not taken: 0.20 is BELOW the live threshold, which throws away the
    # stricter-than-live argument below for three passes on one day.
    #
    # RECALIBRATED 2026-09-08 ON PRODUCTION: 0.30 -> 0.21.
    #
    # 0.30 was chosen on the reasoning that a whole-pass template is a stronger
    # query than one frame, so it could be held to a HIGHER bar than the live
    # 0.215. The reasoning was sound and the number was wrong, because it
    # ignored which gallery the query is compared against.
    #
    # THE GALLERY IS TWO POPULATIONS. 270 of its 328 rows are webcam/ID
    # enrolment photographs; 58 are corridor CCTV crops added by
    # `app/services/augment.py`, covering 34 of 56 people. Measured over 371
    # labelled CCTV passes from this corridor:
    #
    #                              p10     median    p90
    #   CCTV query vs WEBCAM row   0.139   0.310    0.505
    #   CCTV query vs CCTV row     0.256   0.518    0.750
    #
    # A CCTV gallery row is worth +0.208 of median similarity. So 0.30 sat
    # almost exactly ON the median of the webcam-only distribution and threw
    # away half of those passes by construction - and the 22 people who have
    # no CCTV row are precisely the ones the second chance exists to recover.
    #
    # LOWERING IT COSTS NOTHING MEASURABLE, because `augment_live_floor` pins
    # the CCTV rows at 0.35 whatever this number is (Gallery._penalty shifts a
    # row down by its own floor minus the threshold). Dropping the global value
    # therefore loosens ONLY the webcam rows - exactly the population that needs
    # it. Measured on 371 labelled passes:
    #
    #     thr    recall   precision   webcam-only recall
    #     0.18   85.7%      98.5%          75.0%
    #     0.21   83.0%      98.4%          66.7%    <- deployed point
    #     0.30   79.2%      98.3%          50.0%    <- previous
    #     0.36   77.9%      99.0%          41.7%
    #
    # (the sweep steps in 0.03; 0.215 sits inside the 0.21 row)
    #
    # ...and over six days of real unnamed passes (bench/group_calibrate.py,
    # 9348 passes; "novel" = the named person appears nowhere else in
    # production's whole day, the shape a false accept takes):
    #
    #     thr    known agree/wrong   unknown recovered   corroborated  novel
    #     0.18       757 / 8              136                128         8
    #     0.21       732 / 8              116                111         5
    #     0.30       672 / 7              103                 99         4
    #
    # ===================================================================
    # CORRECTION, 2026-09-08, SAME DAY: this was moved 0.30 -> 0.215 on the
    # recall evidence below and then MOVED BACK, because the recall evidence
    # was read at the wrong base rate. The tables are kept in full: they are
    # correct, and they are what a threshold looks like when it is chosen from
    # the wrong denominator.
    #
    # WHAT WAS MISSING: A FALSE ACCEPT RATE. "Wrong" in the tables below is one
    # ENROLLED person matched as another; "novel" is a recovered name whose
    # person appears nowhere else that day. Neither is FAR. FAR is a person who
    # is NOT in the gallery being given a name, and it is the error that
    # corrupts attendance.
    #
    # MEASURED with leave-one-person-out over 902 labelled CCTV passes: each
    # probe's own person is deleted from the gallery, so any name returned is a
    # true false accept (bench/tracklet_far_eval.py).
    #
    #     thr      FPIR    genuine recall
    #     0.18    2.66%        83.9%
    #     0.215   1.22%        80.6%
    #     0.24    1.11%        78.0%
    #     0.27    0.89%        76.2%
    #     0.30    0.78%        74.5%
    #
    # THE PROBE SET IS BALANCED AND PRODUCTION IS NOT. That table has one
    # impostor per genuine probe. This corridor has about FIFTEEN: over six
    # days, 1609 unnamed passes carry a usable face and only ~100 of them are a
    # recoverable employee. The recall gain applies to the small pool and the
    # FPIR gain to the large one, so the trade inverts:
    #
    #     from 0.30    extra genuine   extra false   genuine per false
    #      -> 0.27         +3.2           +1.8             1.8
    #      -> 0.24         +3.7           +5.3             0.7
    #      -> 0.215        +2.9           +7.1             0.4
    #      -> 0.18         +2.8          +30.2             0.1
    #
    # Genuine recoveries SATURATE around 94 whatever the threshold; below 0.27
    # the only thing a lower bar buys is wrong attendance rows. This is the
    # same base-rate argument app/services/pseudo_gallery.py makes for never
    # naming anybody by body, and it was not applied here until FAR was
    # actually measured.
    #
    # 0.30 rather than 0.27: 0.27 is defensible at 1.8 genuine per false, but
    # only if a wrong row costs less than 1.8 missed ones. Everything else in
    # this system is built on a false accept costing FAR more than a miss - a
    # miss is seen again on the next pass, a wrong name is not revealed by
    # anything in the data - so the ratio has to be much larger than that to be
    # worth taking.
    #
    # THE REAL FIX FOR THE DOMAIN GAP IS NOT THIS NUMBER. It is giving the
    # people who have only webcam enrolment photographs a CCTV gallery row -
    # `scripts/augment_gallery.py`. Those rows carry their own 0.35 floor, so
    # they raise recall for exactly the people who need it WITHOUT loosening
    # anything for anybody else, which is the property this threshold does not
    # have.
    # ===================================================================
    #
    # The argument for 0.215, kept because it is where the reasoning went wrong:
    # 0.215 rather than 0.18: it is EXACTLY the live per-frame bar for this
    # recognizer (`recognizer_thresholds["adaface_ir101_finetune_fp16.onnx"]`),
    # so the second chance never accepts a similarity the live matcher would
    # reject. All it adds is whole-pass aggregation and the runner-up margin -
    # a claim that needs no calibration of its own to defend, which matters for
    # a rule that writes attendance. 0.18 is measured, is better on recall
    # (+20 recovered passes over six days for +3 novel names), and is there to
    # take; it just cannot be justified without leaning on this corpus alone.
    #
    # IF THE RECOGNIZER CHANGES, THIS MOVES WITH IT. It is written as a literal
    # rather than read from `threshold_for()` so that a model swap cannot
    # silently re-point it - the same reason `recognizer_thresholds` is a table
    # and not a formula - but it is not independent, and a new recognizer needs
    # both re-derived together.
    #
    # Precision was 98-100% in EVERY bucket of face size and evidence volume at
    # 0.18, including passes with an inter-pupil distance under 20 px. The only
    # errors, five of them, were passes whose crops contain TWO PEOPLE walking
    # together - inspected by eye - and they score 0.42-0.54, so they are
    # present at every threshold and are a track-purity problem, not this one.
    #
    # Empty/0.0 disables the second chance entirely.
    tracklet_face_threshold: float = 0.30
    # The runner-up PERSON must be beaten by this much, exactly as in the live
    # matcher and for the same reason - a top score alone does not distinguish
    # "this person" from "somebody who resembles two people equally".
    tracklet_face_margin: float = 0.05
    # A template built from a single frame is that frame, and the live path has
    # already judged it. Two is the smallest number that makes this a different
    # question from the one already answered.
    tracklet_min_face_frames: int = 2

    # ---- pseudo-person grouping (unregistered people) --------------------
    # A person who is not enrolled still walks past repeatedly. Without this
    # every one of their passes is an independent "unknown" and the system
    # cannot say whether it saw one visitor five times or five visitors once.
    #
    # So an unnamed pass is attached to a stable pseudo-identity (P-000123)
    # built from its FACE and BODY features, and the next pass of the same
    # person joins the same one.
    #
    # VALIDATED ON PRODUCTION, held-out days Thu 09-03, Sun 09-06, Mon 09-07,
    # each setting run ONCE, labelled passes scored inside the day's real crowd
    # (bench/group_distractor_eval.py):
    #
    #                  precision  recall    F1    count error  pseudo-people
    #   0.35 / 0.75      88.0%    51.0%   64.3%      +103%         2963
    #   0.30 / 0.60      84.6%    70.7%   76.1%       +55%         1860
    #
    # Over those three days: 4949 unknown passes become 1860 pseudo-people
    # instead of 2963 - 62% fewer rows for an operator to review rather than
    # 40% - and the distinct-person count is roughly twice as accurate.
    #
    # The research figures below are kept because they are what the design came
    # from; the numbers above are what this deployment measured.
    #
    # MEASURED over three days and ~4200 unnamed passes
    # (integration/docs/INTEGRATSIYA.md, section 3.3):
    #
    #   mode          pair precision  pair recall  pseudo-people  count error
    #   body alone        91-94%        3.5-7.6%     563/803/607    126-198%
    #   face alone        90-98%        23-31%      999/1240/1037     9-23%
    #   face + body       90-98%        29-32%       499/695/520      8-26%
    #
    # Which is why BOTH are used and why FACE LEADS. Body alone cannot regroup
    # anybody: two strangers in similar clothing out-score one person seen
    # twice. Face is clothing-independent and works across days; body's whole
    # contribution is linking the passes that have no usable face at all, and
    # it halves the number of pseudo-people by doing so.
    #
    # Honest limit: recall is ~30%, so one person's passes still split into
    # about three groups. The cause is not the model - only 23-28% of passes
    # contain a face with an inter-pupil distance of 20 px or more. That is a
    # camera-geometry ceiling, and it is why this is good enough for counting
    # distinct visitors and NOT good enough for attendance.
    pseudo_person: bool = True
    # Attaching a pass to an existing pseudo-person by FACE.
    #
    # ===================================================================
    # CORRECTION, 2026-09-08, SAME DAY: moved 0.35 -> 0.30 and then BACK.
    # The calibration below is real but it was run on the wrong features, and
    # the record of that is more useful than a clean-looking number.
    #
    # IT WAS CALIBRATED ON STORED 320-px BODY CROPS. Production does not build
    # face templates from those: `CameraWorker` builds them from the FULL 4K
    # frame on the capture thread. Replaying real footage through the real
    # pipeline (bench/replay_prod_day.py) showed how far apart those are:
    #
    #     usable face (ipd >= 20) in stored 320-px crops :  23.5% of passes
    #     usable face in real 4K frames                  :  72%   of passes
    #
    # Three times the coverage and much higher quality, so the similarity
    # distributions the threshold sits in are not the same distributions. A
    # threshold fitted to degraded features lands too LOW, and it did.
    #
    # ON 4K FEATURES the grouping sweep is flat in precision across face
    # 0.30-0.40 (82.6-83.3%) and the count accuracy improves as the threshold
    # rises. That comparison is circular with respect to the face threshold -
    # the truth is face clusters at 0.45 - so it cannot SELECT the value, and
    # it cannot confirm 0.30 either. With the evidence for 0.30 undermined and
    # nothing production-quality to replace it, this goes back to the
    # research-validated 0.35. Grouping writes no attendance, so the cost of
    # being conservative here is a slightly higher visitor count, not a wrong
    # timesheet.
    #
    # TO SETTLE IT PROPERLY: the replay needs to keep the face templates of the
    # passes it NAMES, which carry real identities and 4K quality at once. That
    # is the one measurement that would be both production-representative and
    # non-circular, and it does not exist yet.
    # ===================================================================
    #
    # The calibration that argued for 0.30, kept for the record:
    # RECALIBRATED 2026-09-08 ON PRODUCTION: 0.35 -> 0.30.
    #
    # This threshold and `tracklet_face_threshold` are NOT on the same scale
    # and must never be tuned together. That one compares a CCTV pass against a
    # gallery of mostly WEBCAM photographs; this one compares a CCTV pass
    # against another CCTV pass, and same-domain pairs score about 0.2 higher
    # (see the note on tracklet_face_threshold). A single number for both would
    # be too strict for one and too loose for the other.
    #
    # HOW IT WAS MEASURED, and why the obvious measurement is worthless. The
    # tempting truth for grouping unknowns is "connected components of face
    # similarity >= 0.45", which is what the research used. Scored that way,
    # this rule is graded against a relabelling of its own output: at a
    # grouping threshold of 0.45 its precision is 100% BY CONSTRUCTION, and
    # every step toward 0.45 looks like progress. Swept on it, the grid asks
    # for 0.43. Swept on real identities it asks for 0.30. Only one of those is
    # measuring anything.
    #
    # So the grid was searched on `data/persons/known/`, where the label is the
    # identity the face path assigned at the time and owes nothing to this
    # rule - and with the whole day's real crowd loaded into the gallery as
    # distractors, because 200 labelled passes over 30 people is not the
    # population a threshold meets in production (bench/group_distractor_eval.py).
    #
    # Calibration days Wed 09-02, Fri 09-04, Sat 09-05 - two weekdays and one
    # weekend day, because this corridor's weekend is a DIFFERENT population
    # (the face path names 12% of passes on a weekday and 2.4% on a Sunday).
    pseudo_face_threshold: float = 0.35
    # ...and by BODY, which is the fallback for a pass with no usable face.
    #
    # ===================================================================
    # CORRECTION, 2026-09-08, SAME DAY: moved 0.75 -> 0.60 and then BACK, on
    # evidence that is NOT circular and is therefore the one to believe.
    #
    # The truth used for grouping is built from FACES only, so comparing two
    # BODY thresholds at a fixed face threshold measures the body rule against
    # a yardstick it had no part in making. Run on real 4K features
    # (bench/replay_prod_day.py, 160 unnamed passes, 46 face-truth people):
    #
    #     face   body   pseudo-people   pair precision
    #     0.30   0.60        50             71.6%
    #     0.30   0.75        67             82.8%
    #     0.35   0.60        56             75.3%
    #     0.35   0.75        75             82.6%
    #
    # Body 0.60 costs ELEVEN POINTS of precision at the same recall - it merges
    # strangers who happen to be dressed alike, which is the exact failure the
    # research says body similarity is prone to. The sweep that recommended
    # 0.60 ran on stored 320-px crops, where only 23.5% of passes carry a
    # usable face against 72% on 4K, so body was carrying far more of the
    # grouping than it ever does in production and its errors were hidden
    # behind a face-poor population.
    # ===================================================================
    #
    # The calibration that argued for 0.60, kept for the record:
    # RECALIBRATED 2026-09-08 ON PRODUCTION: 0.75 -> 0.60.
    #
    # 0.75 was carried over from the research corpus on the argument that this
    # matches against every pseudo-person alive today rather than one other
    # camera's handful, so the impostor pool is larger. The argument is right
    # and 0.75 was still far too strict: with the day's real crowd as
    # distractors, F1 peaks at 0.60 and precision only starts falling below it.
    #
    #   face 0.30, distractor-aware, calibration days:
    #     body   precision  recall    F1     count error
    #     0.45     79.7%    83.8%   81.5%       +17%
    #     0.50     81.5%    83.9%   82.5%       +19%
    #     0.55     84.4%    80.0%   82.0%       +26%
    #     0.60     88.8%    80.0%   84.2%       +33%   <- peak
    #     0.65     89.3%    77.6%   83.0%       +50%
    #     0.75     89.2%    72.3%   79.8%       +74%
    #
    # Every setting still OVER-splits (positive count error), so this is not a
    # trade of counting accuracy against safety - 0.60 is better on both.
    pseudo_body_threshold: float = 0.75
    pseudo_body_margin: float = 0.05
    # "A usable face" - inter-pupil distance in pixels on the source frame.
    # Below this the face embedding is a source of WRONG links rather than a
    # weak one, so it is discarded and the pass falls through to body.
    pseudo_face_ipd_min: float = 20.0
    # Templates kept per pseudo-person, per modality: a ring buffer, newest
    # wins. One averaged template blurs a person seen in two outfits; keeping
    # every template makes the store grow without bound.
    pseudo_templates_k: int = 5
    # A pseudo-person's BODY templates are discarded when the business date
    # rolls over, because clothing changes overnight and a stale body template
    # links two different people wearing the same coat. Face templates are
    # kept: a face is the same face tomorrow.
    pseudo_body_same_day_only: bool = True
    # How far back the pseudo gallery looks for candidates. A visitor who has
    # not appeared in this many days starts a new pseudo-identity rather than
    # keeping the matching cost growing forever.
    pseudo_active_days: int = 30
    # Body crops live under data/, NOT media/. media/ is a public StaticFiles
    # mount, and these are images of unidentified people.
    persons_dir: Path = ROOT / "data" / "persons"

    # 960, not 1280: detector recall is unchanged (40/40 on the gallery, and
    # YOLOv8n-face holds 100% recall down to a 16 px face — see
    # bench/detect_resolution.py), while both the downscale and the forward
    # pass get cheaper.
    detect_width: int = 960        # downscale for detection; crops come from full-res
    detect_conf: float = 0.35
    # The DFA graph resizes its input to 160 regardless, so a larger crop_size is
    # a resample for nothing: measured d-prime is 12.37-12.53 across 144..320.
    align_crop_size: int = 160
    # 1.30 agrees with three independent lines of evidence: the measured sweep
    # (flat 1.0-1.35, falling from 1.43), the break-even arithmetic (160/1.43 =
    # 112, the output size), and the CVLFace authors' own reference input, whose
    # face fills 57% of the frame. See app/core/aligner.py for the full note.
    align_margin: float = 1.30
    # "sharp" warps the 112 crop from the native-resolution crop instead of the
    # aligner's internal 160 tensor - one resample instead of two.
    # Gallery: d-prime 12.53 -> 12.64, weakest genuine pair 0.547 -> 0.571.
    # "reference" restores bit-exact CVLFace PyTorch numerics.
    align_mode: str = "sharp"
    embed_batch: int = 8

    # Hard ceiling on GPU memory per ONNX Runtime session, in MB. 0 = no limit.
    # gpu6 shares one RTX 3090 with several other projects, and a training job
    # there routinely holds 20+ GB; with the card nearly full the pipeline died
    # at the first inference on CUBLAS_STATUS_ALLOC_FAILED - out of room for
    # cuBLAS's workspace, not for weights. Setting this makes the failure
    # predictable and keeps this process from creeping into memory another
    # project is about to want. See app/core/onnx_env.best_providers.
    gpu_mem_limit_mb: int = 0

    # ---- quality gates --------------------------------------------------
    # Calibrated against bench/degrade_test.py, which degraded known gallery
    # faces to corridor conditions and re-matched.  The recognizer held up far
    # better than the original guesses assumed:
    #     60 px face          -> 0.99
    #     sharpness 78 (k=5)  -> 0.96
    #     sharpness 43 (k=9)  -> 0.71-0.84
    #     35 deg pitch        -> 0.91-0.95
    # The first walkthrough gated 104 of 116 observations, mostly on blur, while
    # scores that low are still perfectly recognizable.  Gates now reject only
    # genuinely unusable frames; the threshold and the consensus vote do the
    # real filtering.
    # Measured against HEAD boxes, which run ~1.33x the face box, so this is
    # about a 42 px face.
    #
    # Was 66. Measured 2026-08-28 by replaying 20 recordings: the face-size gate
    # was rejecting 5 575 of 7 578 frames that had a head box - by far the
    # largest loss in the pipeline, dwarfing aligner (418), yaw (54) and blur
    # (40) put together. A 26-second pass embedded 12 of its ~520 frames.
    #
    # CAREFUL - the sweep behind this was run on RECORDINGS, and
    # `record_width = 1920` halves them. The live cameras are 3840x2160, so the
    # same person at the same distance yields a head box TWICE the size in
    # production as in a clip. The sweep's 56 is a 1080p number; its 4K
    # equivalent is ~112.
    #
    # On the recordings, 66 rejected 5 575 of 7 578 head-bearing frames and
    # dropping it to 56 recognised one more person, raised evidence per pass 43%
    # and lifted the mean committed score 0.324 -> 0.358. None of that transfers
    # to 4K.
    #
    # What the LIVE data says (179 recognised passes, real 4K): face_px runs
    # min 66, p25 76, median 89, p90 250. The minimum is exactly the old gate,
    # because the gate set the floor - and 56 would reject 0% of them. On 4K
    # this threshold is barely binding.
    #
    # 56 is kept because it cannot reject anything 66 accepted, and it admits
    # the 56-66 px band that was previously refused. But it is close to a no-op
    # in production, and the recording-derived gains above must not be quoted as
    # production gains. Re-derive it from live face_px, not from clips.
    min_face_px: int = 56
    min_laplacian_var: float = 25.0
    # The DFA aligner's own confidence that it found a face - and the only gate
    # that can tell a face from the BACK OF A HEAD. It is not a face detector:
    # given a hairline it happily regresses symmetric landmarks, which then
    # produce a confident-looking "yaw 6, pitch 3" from meaningless points.
    # Sharpness does not help either - hair texture scored 473 where real faces
    # score 50-270.
    #
    # 0.40 was far below anything that discriminates. Measured:
    #   enrolment faces (n=220)   min 0.9929, none below 0.99
    #   live crops <0.70          backs and tops of heads, no usable face
    #   live crops 0.70-0.90      steep top-down, face barely visible
    #   live crops 0.90-0.99      face visible, looking down
    #   live crops >=0.99         clear near-frontal faces
    #   the reported false accept 0.604 -> matched a real person at 0.219
    # Over 174 live crops, 12 above-threshold matches came from sub-0.90 crops,
    # the worst being aligner 0.545 matching at 0.274 - higher than several
    # genuine recognitions that day.
    #
    # 0.90 keeps every crop with a visible face and drops all 12. It costs
    # recognition opportunities on people walking head-down, which is the right
    # trade: a miss costs nothing because they are seen again, while a false
    # accept records one person as another and nothing in the data reveals it.
    min_aligner_score: float = 0.900
    max_yaw_deg: float = 45.0
    max_pitch_deg: float = 40.0

    # ---- recognition ----------------------------------------------------
    # CALIBRATED 2026-08-19 from two real walkthroughs (bench/calibrate_threshold.py):
    #   not-enrolled walker : 43 gate-passing obs, best score max 0.184
    #   enrolled walkers    : 87 gate-passing obs, best score max 0.514
    # 0.26 sits 41% above the worst impostor ever observed live, and the
    # consensus vote requires vote_min_recognitions such frames to agree.
    # Zero false commits in simulation at every threshold from 0.22 up.
    # The gallery-photo measurement suggested 0.40; live crops are a lower-
    # scoring domain, which is exactly why this had to be measured.
    # PROVISIONAL for adaface_ir101_finetune. Thresholds do not transfer between
    # models: this one compresses impostor scores hard (gallery max 0.208 vs the
    # base model's 0.392), so the whole scale sits lower. 0.14 is the base
    # deployment threshold rescaled by the same fraction of each model's
    # gallery FAR=0 point (0.26/0.395 = 0.66; 0.210 x 0.66 = 0.138).
    # Confirm against a live walkthrough before trusting it.
    # 0.18, raised from 0.14 after a confirmed false positive at 0.157
    # (a forehead crop matched to the wrong person). The lowest accepts are
    # where the errors live: today's accepts ran 0.157-0.522, and the single
    # known error sat at the very bottom. 0.18 drops 4 of 43 current accepts.
    #
    # For attendance a false accept is far worse than a miss - one person is
    # recorded as another, and nothing in the data reveals it - while a miss
    # costs nothing, because the person is seen again on their next pass.
    # Still provisional: it needs a labelled set, not one confirmed error.
    recognition_threshold: float = 0.18
    # Live scores are compressed (0.26-0.53) vs gallery scores (~0.90), so a
    # fixed margin is proportionally weaker here.  0.08 would have excluded the
    # one weak event seen in the first live run (margin 0.065).
    second_best_margin: float = 0.045
    vote_window: int = 5
    vote_required: int = 3

    # Identity is decided by CONSENSUS over the whole pass, not by the first few
    # frames to agree.  A track commits a person when that person accounts for
    # at least `vote_consensus` of the frames that produced ANY identity, and at
    # least `vote_min_recognitions` frames agree on them.
    #
    # Misses are deliberately NOT in the denominator: a frame that matched
    # nobody says nothing about which of two candidates is right, and counting
    # them would punish a person who was turned away for most of a pass.
    #
    # The floor exists because a fraction alone cannot judge a short pass - one
    # agreeing frame out of one is 100% consensus and would clear any percentage
    # rule.  Below the floor the pass commits nobody and is recorded as an
    # unknown sighting, which is the FAR-favouring choice: better to miss a
    # quick walk-through than to name the wrong person from two frames.
    # Thresholds are calibrated PER RECOGNIZER and are not transferable. Swapping
    # the model without swapping the threshold is a silent false-accept
    # generator: each model puts its impostor distribution on a different scale,
    # so a threshold that is safe for one can accept nearly anybody on another.
    #
    # `threshold_for()` resolves it from the model actually loaded, so the two
    # cannot drift apart. An unknown model falls back to recognition_threshold
    # and logs a warning rather than guessing.
    # CALIBRATED 2026-08-31 on native-4K corridor clips (bench/calibrate_threshold.py,
    # 10 clips, 475 gate-passing faces of one enrolled person, each scored against
    # all 54 enrolled identities = ~25,000 impostor comparisons):
    #
    #   genuine median   0.350
    #   impostor MAX     0.195     <- the number that matters
    #   FAR=0 threshold  0.215
    #   genuine frames kept at that threshold: 87%
    #
    # THE PREVIOUS VALUE ACCEPTED OBSERVED IMPOSTORS. 0.18 sat below the worst
    # corridor impostor of 0.195, so the system could - and did - name the wrong
    # person; two such errors were reported from live traffic.
    #
    # Set above every impostor ever observed rather than at an equal-error
    # point, because the two failures are not symmetric: a false accept records
    # one person as another and nothing in the data reveals it, while a miss
    # costs nothing - the person is seen again on their next pass.
    #
    # STILL PROVISIONAL on the genuine side: the impostor statistics are strong
    # (~25,000 pairs), the genuine statistics come from one person's ten passes.
    # Widen it with clips of more enrolled people before treating the genuine
    # retention figure as settled.
    # Overrides the per-model value below when > 0. For tuning on a deployed
    # machine, which has no source tree and cannot rebuild the compiled core:
    #
    #     echo "recognition_threshold_override=0.19" >> .env   # then restart
    #
    # It logs a warning every time it is consulted, because a number set by
    # hand and forgotten is exactly how the deployed 0.18 came to sit below the
    # worst impostor this corridor actually produces. 0 = use the calibration.
    recognition_threshold_override: float = 0.0

    recognizer_thresholds: dict = {
        "adaface_ir101_finetune_fp16.onnx": 0.215,
        "adaface_ir101_finetune.onnx": 0.215,
    }

    # -- augmentation: per-crop acceptance floors ----------------------------
    # A corridor crop added to the gallery does NOT inherit the global
    # threshold. It is measured against every other person's vectors we hold -
    # enrolment photographs and other people's corridor crops alike - and gets
    # its own floor just above the worst of them. Below that floor the crop
    # cannot name anybody, so its false-accept rate against everything we have
    # ever seen is zero BY CONSTRUCTION rather than by hoping the global
    # threshold happens to cover it.
    #
    # This replaced an all-or-nothing gate that refused the whole selection
    # whenever the gallery's worst pair reached the threshold. That pair was
    # two ENROLMENT photographs - Narmatov/Qo'shmatov at 0.202 locally, 0.208 on
    # the server against a 0.190 threshold - so the gate was permanently shut by
    # a defect no selection could fix, while blaming whichever crop was chosen.
    #
    # THE FLOOR EVERY CORRIDOR CROP GETS. Measured on gpu6, 04-09-2026, against
    # 866 real corridor faces the system had judged as nobody - the 472 from
    # 02-09 predate the crops entirely, so they are genuinely out of sample:
    #
    #   floor   genuine live matches kept   unknown faces given a name
    #   0.22            100.0 %                    21.71 %   <- no floor
    #   0.30             95.1 %                     3.54 %
    #   0.35             89.3 %                     1.94 %   <- this
    #   0.40             82.2 %                     1.14 %
    #   0.50             59.6 %                     0.34 %
    #
    #   the same probes reaching an ENROLMENT photograph:  2.0 - 3.3 %
    #
    # 0.35 is the point where a corridor crop becomes SAFER than the studio
    # photograph beside it, while keeping 89% of the recognition it was added
    # for. That is the criterion, and it is the one to re-measure: a crop must
    # never be the weakest row in the gallery.
    #
    # This replaced a per-row floor calibrated from each crop's own worst
    # impostor. That measured the wrong population - the stored gallery, which
    # is studio photography - so every floor came out at the global threshold
    # and all 48 deployed crops were inert. Calibrating per row against real
    # corridor faces instead was still worse than one flat number on BOTH axes
    # (85.3% kept / 5.03% named): a row's maximum over a few hundred probes is
    # a noisy extreme-value estimate, and it over-floors some rows while
    # missing the lookalike who simply did not walk past that day.
    #
    # Re-measure with bench/far_live_rows.py. It is a policy, not a constant.
    augment_live_floor: float = 0.35
    # Still used, but only to REFUSE a crop: one whose worst corridor impostor
    # plus this margin exceeds the floor above is a known lookalike and is not
    # offered at all, rather than being given a bespoke floor of its own.
    augment_threshold_margin: float = 0.02
    # A crop needing a floor above this is not refused for being dangerous - it
    # is refused for being useless. Two corridor crops of the same person under
    # the same lighting land far above this, so a crop that must clear it to be
    # safe would essentially never fire, and would sit in the gallery looking
    # like coverage it does not provide.
    augment_max_threshold: float = 0.60

    vote_consensus: float = 0.65
    vote_min_recognitions: int = 5


    # ---- tracking -------------------------------------------------------
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.2
    track_match_thresh: float = 0.8
    # Frames, not seconds: ByteTrack keeps a lost track for
    # frame_rate/30 * track_buffer frames. At track_frame_rate=20 and
    # track_buffer=36 that is 24 frames = 1.2 s of occlusion tolerance, which
    # is what the pipeline had at 10 fps with the old 12/30 defaults.
    track_frame_rate: int = 20
    track_buffer: int = 36
    track_max_age_s: float = 3.0

    # ---- stream ---------------------------------------------------------
    rtsp_transport: str = "tcp"
    # Every frame. The old value of 2 dated from 12 fps cameras sharing the GPU
    # with a training job; both premises are gone and the streams now run 20 fps.
    # Detection is 22.7 ms against a 50 ms budget per camera at 20 fps, and full
    # rate doubles trajectory density - which is what the 55% UNKNOWN direction
    # rate was starving on (min_points=5 rarely reached at 10 fps).
    process_every_nth: int = 1
    stale_after_s: float = 5.0

    # Replay recorded clips instead of connecting to the cameras. Point it at a
    # directory holding one folder per camera NAME, e.g.
    #   data/recordings_4k/Entrance/*.mp4
    #   data/recordings_4k/Exit/*.mp4
    # and every worker reads its own folder on a loop, paced to real time.
    #
    # This is not a simulation: scripts/record_4k.py stream-copies the camera's
    # own bitstream at native 3840x2160, so the pipeline sees exactly the frames
    # the camera sent. It is how the pipeline can be exercised - and the
    # dashboard demonstrated - from outside the office LAN, where the cameras
    # are unreachable.
    #
    #   replay_dir=data/recordings_4k python scripts/run.py
    replay_dir: Path | None = None
    # Clips are replayed at their ORIGINAL relative times so the two cameras
    # stay in step - one walk really was recorded by both, 1-4 s apart, and
    # back-to-back playback would turn those pairs into unrelated events and
    # never exercise the cross-camera fusion. Idle gaps are capped here: the
    # recordings span an hour of a mostly empty corridor and the dead time
    # carries nothing. Anything shorter than this is preserved exactly.
    replay_max_gap_s: float = 8.0
    # 2 slots dropped ~8% of frames to burst spill once the pipeline ran every
    # frame. The queue is drop-oldest, so extra slots only fill when the
    # pipeline is momentarily behind - at a 15 ms median against a 50 ms budget
    # it sits near-empty, and the worst case costs 2 frames (~100 ms) of
    # staleness on a track that lasts seconds.
    frame_queue_size: int = 4

    # ---- attendance -----------------------------------------------------
    event_cooldown_s: int = 90

    # One physical walk past this corridor is seen by BOTH cameras, and their
    # mirrored geometry makes them disagree about its direction. A completed,
    # identified track is therefore held this long; anything else for the same
    # person inside the window belongs to the same walk, and only the strongest
    # of them moves attendance state. See app/services/arbiter.py.
    #
    # 15 s, from live measurement: the two halves of one walk arrived 2-4 s
    # apart on 2026-08-31 (eight of eight walk-throughs) and 0.6-5.2 s apart in
    # the 2026-08-26 database. 15 s covers that with room for a track that ends
    # late, and is far short of a genuine leave-and-return.
    cross_camera_window_s: float = 15.0

    day_boundary_hour: int = 4
    timezone: str = "Asia/Tashkent"
    open_interval_policy: str = "flag"       # flag | close_at_eod

    # The end-of-day sweep runs INSIDE the service now. It used to exist only in
    # scripts/maintenance.py, driven by a systemd timer that was never installed
    # on either host - so it had never run. 21 rows still claimed people were
    # inside on dates up to five days old.
    eod_sweep_enabled: bool = True
    # Minutes after `day_boundary_hour` to sweep, so a pass at 03:59 has landed.
    eod_sweep_offset_min: int = 10

    # ---- debug ----------------------------------------------------------
    # When on, every recognition writes the full annotated frame, the native
    # face crop and the aligned 112x112 the recognizer actually saw, plus a JSON
    # sidecar - so a wrong name can be inspected instead of guessed at.
    debug_capture: bool = True
    debug_capture_unknown: bool = True     # also keep gate-passing non-matches
    # How many captures each person KEEPS. A rolling window: at the cap the
    # oldest is deleted to make room, so the folder is bounded and the newest
    # recognition always has evidence. Keeping the first N instead bounds it
    # just as well and leaves every later recognition with no picture on the
    # events page, which is where a wrong check-out is judged. 0 means
    # unbounded, which is right for a capture run and wrong for an install
    # that runs for months - the directory is full of face crops.
    debug_max_per_person: int = 40
    debug_dir: Path = ROOT / "data" / "debug"

    # ---- recording (DEBUG ONLY - off by default) -------------------------
    # Clip recording exists to build a replay corpus while tuning: it is how
    # the detector A/B, the direction comparison and the fp16 validation were
    # all measured with the cameras switched off.
    #
    # It is NOT for production. One day of two cameras produced 1,156 clips and
    # 3.2 GB, and those clips are video of identified people - a customer site
    # accumulating that silently is a privacy problem as much as a disk one.
    #
    # Turn it on for a debugging session:  record_clips=1 python scripts/run.py
    record_clips: bool = False
    record_width: int | None = 1920      # None = native 4K
    record_pre_roll_s: float = 2.0
    record_post_roll_s: float = 2.0
    record_max_clip_s: float = 60.0
    record_min_free_gb: float = 10.0

    # Save every gate-passing frame of every track, not just the best one.
    # Debug only, same reasoning as record_clips: it wrote 15,724 images in a
    # day. The best-shot capture below is what a production install keeps.
    save_all_frames: bool = False

    # ---- track tracing (DEBUG ONLY) --------------------------------------
    # Normally only gate-passing frames are embedded, and nothing is saved
    # per frame, so you cannot see how the score behaved across the WHOLE
    # pass - including the frames the quality gates rejected.
    #
    # With this on, every frame of a track is embedded and scored for as long
    # as the track lives, and each aligned face is written with its score. A
    # 5-second pass at 20 fps yields ~100 images named by score, so the
    # approach/retreat curve can be read straight off the filenames.
    #
    # Expensive - it embeds frames the pipeline would otherwise skip - so it
    # stops itself after debug_trace_limit recognised passes.
    debug_trace_tracks: bool = False
    debug_trace_limit: int = 10       # recognised passes, then tracing stops
    save_all_min_free_gb: float = 8.0

    # ---- retention ------------------------------------------------------
    snapshot_retention_days: int = 90

    # ---- api ------------------------------------------------------------
    # Served under a sub-path by a reverse proxy, e.g. /faceid. The edge proxy
    # in front of aiscan.airi.uz does NOT strip the prefix - the backend
    # receives /faceid/... verbatim - so FastAPI's root_path is not enough and
    # every route, static mount, link and WebSocket URL must carry it.
    # Empty means served at the root, which is how it runs locally.
    # Extra hostnames this deployment answers to, comma-separated. Only read
    # when a request carries no Sec-Fetch-Site header (a browser older than
    # Chrome 76 / Firefox 90 / Safari 16.4), where the cross-site check falls
    # back to comparing the Origin against the Host - which a proxy rewrites.
    # Set it to the public name if such a browser has to be supported:
    #     trusted_hosts=aiscan.airi.uz
    trusted_hosts: str = ""

    url_prefix: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = Field(default="change-me-in-production")

    def threshold_for(self, model_name: str | None = None) -> float:
        """The recognition threshold calibrated for the recognizer in use.

        Thresholds do not transfer between models - see recognizer_thresholds.
        Resolved from the model rather than read from a single global so that
        changing `recognizer_model` cannot silently leave the old threshold in
        place.

        `recognition_threshold_override` beats everything, so the value can be
        tuned from `.env` on a machine that has no source tree. Without it the
        only way to change the operating point was to edit this file and
        rebuild the compiled core - which on the server is not possible at all.
        """
        import logging
        if self.recognition_threshold_override > 0:
            logging.getLogger(__name__).warning(
                "recognition_threshold_override=%.3f is in force; the value "
                "calibrated for this recognizer is being ignored. Remember to "
                "clear it once tuning is done.",
                self.recognition_threshold_override)
            return float(self.recognition_threshold_override)
        name = Path(str(model_name or self.recognizer_model)).name
        name = name.replace(".enc", "")
        if name in self.recognizer_thresholds:
            return float(self.recognizer_thresholds[name])
        logging.getLogger(__name__).warning(
            "no calibrated threshold for recognizer %r; falling back to %.3f. "
            "Thresholds are NOT transferable between models - calibrate before "
            "trusting this.", name, self.recognition_threshold)
        return float(self.recognition_threshold)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


def camera_credentials() -> tuple[str, str]:
    """Camera login for the ops scripts that talk to cameras directly.

    From the environment, never from source. Four scripts had the password
    written in as a literal - a working credential for a live camera in every
    copy of the file, and in every clone once this repository became public. An
    audit before the first push caught it.

    The running service does not use this: it reads the whole RTSP URL from the
    `camera` table, credentials included.

        export CAMERA_USER=admin CAMERA_PASSWORD='...'
    """
    import os
    pwd = os.environ.get("CAMERA_PASSWORD", "")
    if not pwd:
        raise SystemExit(
            "  CAMERA_PASSWORD is not set. This script connects to the cameras "
            "directly:\n"
            "    export CAMERA_USER=admin CAMERA_PASSWORD='<password>'")
    return os.environ.get("CAMERA_USER", "admin"), pwd


settings = Settings()
