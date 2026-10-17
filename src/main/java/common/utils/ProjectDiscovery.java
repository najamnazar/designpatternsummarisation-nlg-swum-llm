package common.utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Shared project discovery and output-directory helpers for the NLG and LLM pipelines.
 * <p>
 * CHANGE (refactor): this logic used to be copied into both {@code dps_nlg.Application} and
 * {@code dps_llm.DpsLlmApplication}. The NLG copy did not sort directories, so the order in
 * which projects were visited came from {@link File#listFiles()}, which the JDK does not
 * guarantee. Because {@code ParseProject} skips files it has already seen (same name and
 * content), that order decided which of two identical folders (e.g. {@code Abstract Factory}
 * and {@code AbstractFactory}) received the CSV rows, and it could differ between machines.
 * Every listing here is now sorted by path so all pipelines visit projects in the same order.
 * </p>
 */
public final class ProjectDiscovery {

    private ProjectDiscovery() {
        // Utility class - prevent instantiation
    }

    /**
     * Recursively finds every directory that directly contains at least one {@code .java} file,
     * in deterministic (path-sorted) order.
     *
     * @param root the directory to search
     * @return project directories sorted by path
     */
    public static List<File> findProjectDirectories(File root) {
        List<File> projectDirs = new ArrayList<>();
        findProjectDirectoriesRecursive(root, projectDirs);
        projectDirs.sort(Comparator.comparing(File::getPath));
        return projectDirs;
    }

    private static void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
        if (directory == null || !directory.isDirectory()) {
            return;
        }

        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        if (javaFiles != null && javaFiles.length > 0) {
            projectDirs.add(directory);
        }

        for (File subdir : sortedChildren(directory, true)) {
            findProjectDirectoriesRecursive(subdir, projectDirs);
        }
    }

    /**
     * Lists the children of a directory sorted by name, so traversal never depends on
     * filesystem order.
     *
     * @param directory the directory to list
     * @param directoriesOnly when true, only sub-directories are returned
     * @return sorted children, or an empty list if the directory cannot be read
     */
    public static List<File> sortedChildren(File directory, boolean directoriesOnly) {
        File[] children = directoriesOnly ? directory.listFiles(File::isDirectory) : directory.listFiles();
        if (children == null) {
            return new ArrayList<>();
        }
        Arrays.sort(children, Comparator.comparing(File::getName));
        return Arrays.asList(children);
    }

    /**
     * Creates each directory (and any missing parents) if it does not exist yet.
     *
     * @param directories directory paths relative to the working directory
     * @throws IOException if a directory cannot be created
     */
    public static void createDirectories(String... directories) throws IOException {
        for (String dirPath : directories) {
            File dir = new File(dirPath);
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IOException("Failed to create directory: " + dirPath);
            }
        }
    }

    /**
     * Deletes the {@code .json} files directly inside a directory.
     * <p>
     * CHANGE: added so a pipeline can clear its previous JSON output before regenerating it.
     * Without this, JSON files from input folders that were later renamed or removed (e.g. the
     * old {@code spring-framework-main_*} files) stayed in {@code output/json-output/nlg} and
     * were picked up again by the SWUM pipeline, which reads every JSON file in that folder.
     * </p>
     *
     * @param directory the directory to clear; nothing happens if it does not exist
     * @return number of files deleted
     * @throws IOException if a file cannot be deleted
     */
    public static int deleteJsonFiles(File directory) throws IOException {
        File[] jsonFiles = directory.listFiles((dir, name) -> name.endsWith(".json"));
        if (jsonFiles == null) {
            return 0;
        }
        for (File jsonFile : jsonFiles) {
            if (!jsonFile.delete()) {
                throw new IOException("Failed to delete stale JSON output: " + jsonFile.getAbsolutePath());
            }
        }
        return jsonFiles.length;
    }
}
