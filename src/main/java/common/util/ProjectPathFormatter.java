package common.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Utility helpers for transforming the raw project identifier that flows through
 * the pipelines into user friendly project and folder names.
 */
public final class ProjectPathFormatter {

    private ProjectPathFormatter() {
        // Utility class
    }

    public static final class Parts {
        private final String projectName;
        private final String folderName;

        public Parts(String projectName, String folderName) {
            this.projectName = Objects.requireNonNullElse(projectName, "").trim();
            this.folderName = Objects.requireNonNullElse(folderName, "").trim();
        }

        public String projectName() {
            return projectName;
        }

        public String folderName() {
            return folderName;
        }
    }

    /**
     * Split the raw identifier (which may contain slashes, underscores or other separators)
     * into a base project name and a folder path.
     */
    public static Parts split(String rawIdentifier) {
        if (rawIdentifier == null || rawIdentifier.isBlank()) {
            return new Parts("", "");
        }

        String normalized = rawIdentifier.trim().replace('\\', '/');
        String projectName = normalized;
        String folderPortion = "";

        int slashIndex = normalized.indexOf('/');
        if (slashIndex >= 0) {
            projectName = normalized.substring(0, slashIndex).trim();
            folderPortion = normalized.substring(slashIndex + 1).trim();
        } else {
            int underscoreIndex = normalized.indexOf('_');
            if (underscoreIndex >= 0) {
                projectName = normalized.substring(0, underscoreIndex).trim();
                folderPortion = normalized.substring(underscoreIndex + 1).trim();
            }
        }

        if (folderPortion.isEmpty()) {
            return new Parts(projectName, "");
        }

        folderPortion = folderPortion.replace('\\', '/');
        // Treat multiple underscores as directory separators when no slashes exist
        folderPortion = folderPortion.replaceAll("_+", "/");

        String[] rawSegments = folderPortion.split("/");
        List<String> cleanedSegments = new ArrayList<>(rawSegments.length);
        for (String segment : rawSegments) {
            if (segment == null) {
                continue;
            }
            String cleaned = segment.trim();
            if (cleaned.isEmpty()) {
                continue;
            }
            // Collapse repeated whitespace inside the segment
            cleaned = cleaned.replaceAll("\\s+", " ");
            cleanedSegments.add(cleaned);
        }

        String folderName = String.join(" / ", cleanedSegments);
        return new Parts(projectName, folderName);
    }
}
