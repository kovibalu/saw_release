# Shading Annotations in the Wild (SAW) Annotation Data Format

## Citation and License

See our paper for details of how this data was collected:

    Balazs Kovacs, Sean Bell, Noah Snavely, Kavita Bala. "Shading Annotations in the Wild".
    Computer Vision and Pattern Recognition (CVPR 2017). http://opensurfaces.cs.cornell.edu/publications/saw/.

Please cite our paper if you find our data useful.  The JSON files in `saw/saw_annotations_json`
are licensed under a [Creative Commons Attribution 4.0 International
License](http://creativecommons.org/licenses/by/4.0/). This directory will be
created when you run the `download_saw.sh` script.

Please let me know if you have any questions: Balazs Kovacs, bkovacs@cs.cornell.edu.


## JSON Data Format

The data format of the original [Intrinsic Images in the Wild](http://opensurfaces.cs.cornell.edu/publications/intrinsic/) paper is extended
with the new type of annotations in SAW. Each photo is indexed by its OpenSurfaces ID,
stored in `saw/saw_images_512/[id].png` and
`saw/saw_annotations_json/[id].json`.

Below is an example judgements JSON file with comments:

```
    {
      # OpenSurfaces photo ID
      "photo": 100010,

      # Image filename (will always be "[id].png")
      "image_filename": "100010.png",

      # Width/height aspect ratio
      "aspect_ratio": 1.33333333333333,

      # Name used to give attribution to the photographer
      # This is empty for images in the NYUv2 dataset
      "attribution_name": "David Shankbone",

      # This is true if the photo is part of the NYUv2 dataset
      "in_nyu_dataset": true,

	  # Index of the image in the NYUv2 dataset. We use this to establish
	  # correspondence between our OpenSurfaces ID and the normal and depth maps
	  # for the NYUv2 images.
	  # It is null for images that are not in the NYUv2 dataset.
      "nyu_idx": 417,

      # URL used to give attribution to the photographer (e.g. flickr URL)
      # This is empty for images in the NYUv2 dataset
      "attribution_url": "http://www.flickr.com/photos/shankbone/3022857685/",

      # The Flickr ID (i.e. the last part of the URL for the photo)
      # This is null for images in the NYUv2 dataset
      "flickr_id": "3022857685",

      # The Flickr username (i.e. the second last part of the URL for the photo)
      # This is null for images in the NYUv2 dataset
      "flickr_user": "shankbone",

      # If not null: this photograph is part of a series of photos with the
      # same lighting condition.  Photos with the same value of light_stack are
      # of the same scene but have different lighting.
      "light_stack": null,

      # License provided by the photographer
      "license": {

        # Is this a creative commons license?
        "cc": true,

        # License name
        "name": "Attribution 2.0 Generic",

        # License URL
        "url": "http://creativecommons.org/licenses/by/2.0/"

      },

      # list of constant shading regions in the image
	  # the format of how the shape is defined is the same as in OpenSurfaces
      "constant_shading_regions": [
      	{
          # Unique ID for this region
          "id": 852150,

		  # Area in normalized units. 1 would mean the region spans the whole
		  # photo.
          "area": 0.00168114301578817,

		  # Area of region in pixels.
		  "pixel_area": 516,

		  # Triangles format- "p1,p2,p3,p2,p3,p4...", where p_i is an index into
		  # vertices, and p1-p2-p3 is a triangle. Each triangle is three indices into points; all
		  # triangles are listed together. This format allows easy embedding into javascript.
		  "triangles": "0,1,2,3,4,5,6,1,0,1,7,2,5,7,1,4,7,5",

		  # Number of triangles, should be equal to len(triangles.split(’,’))//3.
		  "num_triangles": 6,

		  # Vertices format- "x1,y1,x2,y2,x3,y3,...",
		  # (coords are fractions of width/height) (this format allows easy
		  # embedding into javascript).
		  "vertices": "0.8058978388235585,0.5739940893964142,0.8091788534020996,0.56087003108225,0.8489611551669094,0.5734472536333239,0.8009763169557469,0.5455586297157252,0.8013864437780646,0.5340750786908316,0.8087687265797818,0.5493864800573565,0.8058978388235585,0.5646978814238812,0.8514219161008152,0.5379029290324628"

		  # Number of vertices, should be equal to len(vertices.split(’,’))//2.
		  "num_vertices": 8,

		  # Segments format- "p1,p2,p2,p3,...", where p i is an index into
		  # vertices, and p1-p2, p2-p3,... are the line segments. The
		  # segments are unordered.
		  # Each line segment is two indices into points; all segments are listed together.
		  # This format allows easy embedding into javascript.
		  "segments": "2,0,5,3,3,4,0,6,6,1,7,2,1,5,4,7",

		  # Number of segments, should be equal to len(segments.split(’,’))//2.
          "num_segments": 8,

		  # If true, then this region is flat and has only one material (this
		  # is obtained by aggregating the user responses with CUBAM).
		  "shading_region_flat": true,

          # How the "shading_region_flat" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
		  "shading_region_flat_method": "C",

          # CUBAM score obtained by analyzing the worker responses.
          # Large positive value: more certain that it is flat and has only one material.
          # Large negative value: more certain that it is not flat or has multiple materials.
          # Value close to 0: unsure answer.
		  "shading_region_flat_score": 1.07342773807969,

		  # If true, then this region is on a glossy surface (this
		  # is obtained by aggregating the user responses with CUBAM).
		  "shading_region_glossy": false,

          # How the "shading_region_glossy" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
		  "shading_region_glossy_method": "C",

          # CUBAM score obtained by analyzing the worker responses.
          # Large positive value: more certain that it is glossy.
          # Large negative value: more certain that it is not glossy.
          # Value close to 0: unsure answer.
		  "shading_region_glossy_score": -0.560218407784514,

		  # If true, then this region has varying shading (this
		  # is obtained by aggregating the user responses with CUBAM).
		  "shading_region_var": false,

          # How the "shading_region_var" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
		  "shading_region_var_method": "C",

          # CUBAM score obtained by analyzing the worker responses.
          # Large positive value: more certain that it has varying shading.
          # Large negative value: more certain that it does not have varying shading.
          # Value close to 0: unsure answer.
		  "shading_region_var_score": -1.13160541824961,
        }
      ],

      # List of points that were sampled in the photo
      "intrinsic_points": [
        {
          # Unique ID for this point
          "id": 852150,

          # Image pixel value at this location, encoded in RRGGBB hex (like
          # HTML color codes)
          "sRGB": "716736",

          # x coordinate, normalized by width (i.e. in range [0, 1)).
          "x": 0.0261149244051588,

          # y coordinate, normalized by height (i.e. in range [0, 1)).
          "y": 0.97325548441313,

          # Radius used for Poisson disk sampling (as fraction of the image
          # diameter)
          "min_separation": 0.07,

          # If true, then this point is neither on a mirror nor on a
          # transparent surfaces (this is obtained by aggregating
          # the user responses (opaque_responses) with CUBAM).
          "opaque": true,

          # How the "opaque" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
          "opaque_method": "C",

          # CUBAM score obtained by analyzing the opaque_responses.
          # Large positive value: more certain that it is opaque.
          # Large negative value: more certain that it is not opaque.
          # Value close to 0: unsure answer.
          "opaque_score": 0.902872954700391,

          # List of individual responses from workers
          "opaque_responses": [

            {
              # Unique ID for this response
              "id": 544299,

              # MTurk worker ID that gave the answer
              "mturk_worker_id": "A28JGET2XTPRG5",

              # The worker's answer
              "opaque": true,

              # Amount of time the user took to perform the task, excluding
              # time when the window was not in focus (time for all points /
              # number of points).
              "time_active_ms": 73,

              # Same as time_active_ms, but including all time spent (even if
              # they switched windows).
              "time_ms": 73
            },

            # other responses for this point
            ...

          ],

          # If true, then this point is on a
          # glossy surface (this is obtained by aggregating
          # the user responses (glossy_responses) with CUBAM).
          "glossy": true,

          # How the "glossy" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
          "glossy_method": "C",

          # CUBAM score obtained by analyzing the glossy_responses.
          # Large positive value: more certain that it is glossy.
          # Large negative value: more certain that it is not glossy.
          # Value close to 0: unsure answer.
          "glossy_score": 0.902872954700391,

          # List of individual responses from workers
          "glossy_responses": [

            {
              # Unique ID for this response
              "id": 544299,

              # MTurk worker ID that gave the answer
              "mturk_worker_id": "A28JGET2XTPRG5",

              # The worker's answer
              "glossy": true,

              # Amount of time the user took to perform the task, excluding
              # time when the window was not in focus (time for all points /
              # number of points).
              "time_active_ms": 73,

              # Same as time_active_ms, but including all time spent (even if
              # they switched windows).
              "time_ms": 73
            },

            # other responses for this point
            ...

          ],

        },

        # other points
        ...

      ],

      # list of comparisons in the image (one comparison is a pair of points)
      "intrinsic_comparisons": [
        {
          # Unique ID for this comparison
          "id": 1679485,

          # ID of point 1 (look up the point in intrinsic_points to find the
          # point info)
          "point1": 852110,

          # ID of point 2
          "point2": 852107,

          # Is this a reflectance or a shading comparison? Possible values:
          #   "R": Reflectance comparison
          #   "S": Shading comparison
          "compare_what": "S",

          # Which point has a darker surface reflectance: 1, 2, E.  "E"
          # indicates that the two points are about the same.
          "darker": "E",

          # How the "darker" field was set.  Possible values:
          #   "C": CUBAM was used for analysis
          #   "A": An author corrected the value
          "darker_method": "C",

          # CUBAM score obtained by analyzing the darker_responses.
          # This is the "weight" in our WHDR metric (see paper).
          "darker_score": 1.2135110345365,

          # Radius used for Poisson disk sampling (as fraction of the image
          # diameter).  In the paper, we use 0.03 and 0.07 as "dense" and
          # "sparse" edges, respectively.
          "min_separation": 0.07,

          # User responses from when we asked them which point was darker
          "darker_responses": [
            {
              # Unique ID for this response
              "id": 985341,

              # Which point has a darker surface reflectance/shading 
              # (see the "compare_what" field): 1, 2, E.  "E"
              # indicates that the two points are about the same.
              "darker": "E",

              # User-reported confidence.  0: Guessing, 1: Probably, 2:
              # Definitely.  This value is not used in any analysis,
              # as users did not reliably report their confidence
              "confidence": 2,

              # MTurk worker ID that gave this answer
              "mturk_worker_id": "A1TQYQG3VFJTWW",

              # Amount of time the user took to respond to this particular
              # judgement, excluding time when the window was not in focus.
              "time_active_ms": 5741,

              # Same as time_active_ms, but including all time spent (even if
              # they switched windows).
              "time_ms": 8502
            },

            # Other responses for this comparison
            ...

          ]
        },

        # Other comparisons
        ...

      ],

      # list of shadow boundary points in the image
      "shadow_boundary_points": [
        {
          # Unique ID for this point
          "id": 3051843,

          # Image pixel value at this location, encoded in RRGGBB hex (like
          # HTML color codes)
          "sRGB": "716736",

          # x coordinate, normalized by width (i.e. in range [0, 1)).
          "x": 0.0261149244051588,

          # y coordinate, normalized by height (i.e. in range [0, 1)).
          "y": 0.97325548441313,

		  # If true, then this point is on a shadow boundary (this is obtained
		  # by aggregating the user responses (osb_responses) with majority
		  # voting). Note that we do not list candidate points which were not
		  # classified as shadow boundaries.
          "osb": true,

          # How the "osb" field was set.  Possible values:
          #   "M": Majority voting
          #   "A": An author corrected the value
          "osb_method": "M",

          # CUBAM score obtained by analyzing the osb_responses.
          # Since we used majority voting in this case, this is always null.
          "osb_score": null,

          # List of individual responses from workers
          "osb_responses": [

            {
              # Unique ID for this response
              "id": 544299,

              # MTurk worker ID that gave the answer
              "mturk_worker_id": "A28JGET2XTPRG5",

              # The worker's answer
              "osb": true,

              # Amount of time the user took to perform the task, excluding
              # time when the window was not in focus (time for all points /
              # number of points).
              "time_active_ms": 73,

              # Same as time_active_ms, but including all time spent (even if
              # they switched windows).
              "time_ms": 73
            },

            # other responses for this point
            ...

          ],
        }
      ],
    }
```
