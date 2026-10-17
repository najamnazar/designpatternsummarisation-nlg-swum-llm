package common.utils;

/**
 * Locations of the input data, relative to the repository root (the working directory the
 * pipelines are run from).
 * <p>
 * CHANGE (input layout): the input folder is now split into
 * <pre>
 * input/
 * ├── corpus/                       Java sources: corpus/&lt;Project&gt;/&lt;Pattern&gt;/*.java
 * └── ground-truth/                 DPS_Human_Summaries.csv
 * </pre>
 * Before, the project folders and the CSV sat directly in {@code input/}. The pipelines must
 * scan {@link #CORPUS_DIR}, not {@link #INPUT_DIR}: project identifiers are taken relative to
 * the scanned root, so scanning {@code input/} would label every row with project
 * {@code corpus}. Python uses the same paths from {@code python/project_paths.py}.
 * </p>
 */
public final class ProjectPaths {

    /** Top-level input folder. */
    public static final String INPUT_DIR = "input";

    /** Root of the Java corpus: {@code <Project>/<Pattern>/*.java}. */
    public static final String CORPUS_DIR = INPUT_DIR + "/corpus";

    /** Folder holding the human reference summaries. */
    public static final String GROUND_TRUTH_DIR = INPUT_DIR + "/ground-truth";

    /** Human reference summaries (used by the Python evaluation scripts). */
    public static final String HUMAN_SUMMARIES_CSV = GROUND_TRUTH_DIR + "/DPS_Human_Summaries.csv";

    private ProjectPaths() {
        // Utility class - prevent instantiation
    }
}
