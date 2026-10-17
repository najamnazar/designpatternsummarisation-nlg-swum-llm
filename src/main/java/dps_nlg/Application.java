package dps_nlg;

import java.io.File;
import java.io.IOException;
// import java.util.ArrayList; // CHANGE: directory discovery moved to ProjectDiscovery
import java.util.HashMap;
import java.util.List;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;

import common.projectparser.ParseProject;
import common.utils.ProjectDiscovery;
import common.utils.ProjectPaths;
import dps_nlg.summarygenerator.Summarise;

/**
 * Main application entry point for NLG-based design pattern summarization.
 * <p>
 * This application processes Java projects using Natural Language Generation (NLG)
 * techniques with the SimpleNLG library to produce human-readable summaries of classes,
 * methods, and design pattern implementations. It recursively discovers Java projects,
 * parses their structure, and generates both JSON output and CSV summaries.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Discover all Java project directories recursively</li>
 *   <li>Parse project structures using the DPS parser</li>
 *   <li>Generate NLG-based summaries for each project</li>
 *   <li>Write structured JSON output and CSV summary files</li>
 *   <li>Track duplicate files and processing statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class Application {

    private static final String NLG_JSON_DIR = "output/json-output/nlg";

    /**
     * Main entry point for the NLG summarization application.
     * <p>
     * Initiates the complete processing workflow including directory setup,
     * project discovery, parsing, and summary generation.
     * </p>
     * 
     * @param args command line arguments (currently unused)
     */
    public static void main(String[] args) {
        try {
            runApplication();
        } catch (IOException e) {
            System.err.println("Fatal error during application execution: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void runApplication() throws IOException {
        ParseProject parseProject = new ParseProject();

        // Create output and reference Directory if non-existent
        createDirectories();

        // CHANGE: clear JSON output from earlier runs. SWUMEvaluationPipeline summarises every JSON file
        // in this folder, so files left over from renamed or removed input folders (e.g. the old
        // spring-framework-main_* files) would otherwise reappear as extra rows in swum_summaries.csv.
        int removed = ProjectDiscovery.deleteJsonFiles(new File(NLG_JSON_DIR));
        if (removed > 0) {
            System.out.println("Removed " + removed + " JSON files from a previous run in " + NLG_JSON_DIR);
        }
        
        // Reset duplicate tracking
        ParseProject.resetDuplicateTracking();

        // Recursively find all directories containing Java files
        // CHANGE (input layout): the Java sources moved to input/corpus/<Project>/<Pattern>/.
        // Scanning input/ would make "corpus" the project name of every row.
        // File inputDir = new File("input");
        // if (!inputDir.exists() || !inputDir.isDirectory()) {
        //     throw new IOException("Input directory not found or is not a directory");
        // }
        File inputDir = new File(ProjectPaths.CORPUS_DIR);
        if (!inputDir.exists() || !inputDir.isDirectory()) {
            throw new IOException("Corpus directory not found or is not a directory: " + inputDir.getAbsolutePath()
                    + " (expected " + ProjectPaths.CORPUS_DIR + "/<Project>/<Pattern>/*.java)");
        }
        
        List<File> projectDirs = findAllProjectDirectories(inputDir);
        System.out.println("Found " + projectDirs.size() + " project directories to process\n");
        
        for (File project : projectDirs) {
            processProject(project, parseProject, inputDir);
        }
        
        // Close the CSV writer to finalize the summary file
        Summarise.closeCsvWriter();
        
        // Report duplicate statistics
        int skippedCount = ParseProject.getSkippedDuplicatesCount();
        System.out.println("\nAll projects processed. CSV summary file has been generated.");
        if (skippedCount > 0) {
            System.out.println("Skipped " + skippedCount + " duplicate files (same name and content).");
        }
    }
    
    /**
     * Recursively find all directories that contain Java files
     * <p>
     * CHANGE: delegates to {@link ProjectDiscovery}, which returns the directories sorted by path.
     * The original version used unsorted {@code listFiles()} order, so which of two identical
     * input folders received the rows (duplicate files are skipped after the first) depended on
     * the filesystem.
     * </p>
     */
    private static List<File> findAllProjectDirectories(File directory) {
        return ProjectDiscovery.findProjectDirectories(directory);
        // Original implementation (unsorted):
        // List<File> projectDirs = new ArrayList<>();
        // findProjectDirectoriesRecursive(directory, projectDirs);
        // return projectDirs;
    }

    // CHANGE: replaced by ProjectDiscovery.findProjectDirectories.
    // private static void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
    //     if (!directory.isDirectory()) {
    //         return;
    //     }
    //
    //     // Check if this directory contains any Java files
    //     File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
    //     boolean hasJavaFiles = javaFiles != null && javaFiles.length > 0;
    //
    //     if (hasJavaFiles) {
    //         projectDirs.add(directory);
    //     }
    //
    //     // Recurse into subdirectories
    //     File[] subdirs = directory.listFiles(File::isDirectory);
    //     if (subdirs != null) {
    //         for (File subdir : subdirs) {
    //             findProjectDirectoriesRecursive(subdir, projectDirs);
    //         }
    //     }
    // }

    private static void createDirectories() throws IOException {
        // CHANGE: dropped "reference", which was created but never used; creation is shared via ProjectDiscovery.
        // String[] directories = {"output", "output/json-output", "output/json-output/nlg", "output/summary-output", "reference"};
        // for (String dirPath : directories) {
        //     File dir = new File(dirPath);
        //     if (!dir.exists() && !dir.mkdirs()) {
        //         throw new IOException("Failed to create directory: " + dirPath);
        //     }
        // }
        ProjectDiscovery.createDirectories("output", "output/json-output", NLG_JSON_DIR, "output/summary-output");
    }
    
    private static void processProject(File project, ParseProject parseProject, File inputDir) throws IOException {
        // Calculate relative path from input directory
        String relativePath = inputDir.toPath().relativize(project.toPath()).toString().replace("\\", "/");
        System.out.println("\n" + relativePath);
        HashMap<String, Object> parsedProject;

        try {
            // Each directory in input folder is parsed with its relative path
            parsedProject = parseProject.parseProject(project, relativePath, true);
        } catch (Exception e) {
            System.err.println("\tError during project " + relativePath + ": " + e.getMessage());
            e.printStackTrace();
            return; // Continue with next project instead of throwing
        }

        ObjectWriter writer = new ObjectMapper()
                .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        if (parsedProject.isEmpty()) {
            System.out.println("\tEmpty");
            return;
        }
        
        // Use relative path for JSON filename (sanitize for filesystem)
        String jsonFileName = relativePath.replace("/", "_").replace("\\", "_");
        writer.writeValue(new File(NLG_JSON_DIR + "/" + jsonFileName + ".json"), parsedProject);
    }
}

