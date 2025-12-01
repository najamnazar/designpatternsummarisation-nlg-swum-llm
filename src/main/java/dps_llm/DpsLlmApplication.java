package dps_llm;

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import common.projectparser.ParseProject;
import dps_llm.client.LlmClient;
import dps_llm.config.DotEnvLoader;
import dps_llm.summary.LlmSummaryService;
import dps_llm.summary.LlmSummaryService.SummaryStats;
import dps_llm.summary.LlmSummaryWriter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main application entry point for generating LLM-based code summaries.
 * <p>
 * This class orchestrates the LLM-based summarization pipeline, which processes Java projects
 * from the input directory, extracts structural features, and generates natural language summaries
 * using a remote Large Language Model API (OpenRouter). The application reads configuration from
 * a .env file, processes projects recursively, and outputs both JSON feature data and CSV summaries.
 * </p>
 * <p>
 * Key responsibilities:
 * <ul>
 *   <li>Load and validate API configuration from .env file or environment variables</li>
 *   <li>Recursively discover Java project directories in the input folder</li>
 *   <li>Parse and extract code features from each project</li>
 *   <li>Generate LLM-based summaries via remote API calls</li>
 *   <li>Write structured output to JSON and CSV files</li>
 *   <li>Track and report processing statistics</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class DpsLlmApplication {

    private static final String LLM_SUMMARY_PATH = "output/summary-output/llm_summaries.csv";
    private static final String DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions";
    private static final String DEFAULT_MODEL = "mistralai/mixtral-8x22b-instruct";

    /**
     * Application entry point.
     * <p>
     * Initializes and runs the LLM summarization pipeline. Catches and reports any fatal errors
     * that occur during execution, exiting with status code 1 on failure.
     * </p>
     * 
     * @param args command line arguments (currently unused)
     */
    public static void main(String[] args) {
        try {
            new DpsLlmApplication().run();
        } catch (IOException e) {
            System.err.println("Fatal IO error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Executes the main LLM summarization workflow.
     * <p>
     * This method orchestrates the complete processing pipeline:
     * <ol>
     *   <li>Creates required output directories</li>
     *   <li>Loads API configuration from .env file</li>
     *   <li>Initializes the LLM client with configuration parameters</li>
     *   <li>Discovers all Java projects in the input directory</li>
     *   <li>Processes each project to generate summaries</li>
     *   <li>Writes results to CSV and JSON files</li>
     *   <li>Reports final statistics</li>
     * </ol>
     * </p>
     * 
     * @throws IOException if directory creation, file I/O, or API communication fails
     */
    private void run() throws IOException {
        ParseProject parseProject = new ParseProject();
        createDirectories();

        // Ensure duplicate tracking starts clean for this run
        ParseProject.resetDuplicateTracking();

        Map<String, String> dotEnv = DotEnvLoader.load(Path.of(".env"));
        if (dotEnv.isEmpty()) {
            System.out.println("No .env file found or it was empty. Falling back to OS environment variables.");
        }
        String apiKey = resolveValue(dotEnv, "OPENROUTER_API_KEY");
        if (apiKey == null) {
            System.out.println("OPENROUTER_API_KEY not set. Populate .env or export an environment variable before running the LLM pipeline.");
            return;
        }

        String apiUrl = resolveValue(dotEnv, "OPENROUTER_API_URL");
        if (apiUrl == null) {
            apiUrl = DEFAULT_API_URL;
            System.out.println("OPENROUTER_API_URL not set. Using default endpoint " + DEFAULT_API_URL);
        }

        String model = resolveValue(dotEnv, "OPENROUTER_MODEL");
        if (model == null) {
            model = DEFAULT_MODEL;
            System.out.println("OPENROUTER_MODEL not set. Using default model " + DEFAULT_MODEL);
        }

        int maxTokens = resolveInt(dotEnv, "OPENROUTER_MAX_TOKENS", 256);
        double temperature = resolveDouble(dotEnv, "OPENROUTER_TEMPERATURE", 0.2);
        String referer = resolveValue(dotEnv, "OPENROUTER_HTTP_REFERER");
        String title = resolveValue(dotEnv, "OPENROUTER_TITLE");

        LlmClient client = new LlmClient(apiUrl, apiKey, model, maxTokens, temperature, referer, title);
        LlmSummaryService summaryService = new LlmSummaryService(client);

        File inputRoot = new File("input");
        if (!inputRoot.exists() || !inputRoot.isDirectory()) {
            System.out.println("No projects found in input directory.");
            return;
        }

        List<File> projectDirs = findAllProjectDirectories(inputRoot);
        if (projectDirs.isEmpty()) {
            System.out.println("No projects found in input directory.");
            return;
        }

        projectDirs.sort(Comparator.comparing(File::getPath));

        int projectLimit = resolveProjectLimit(dotEnv);
        if (projectLimit > 0 && projectLimit < projectDirs.size()) {
            System.out.printf("LLM_PROJECT_LIMIT=%d set. Processing first %d projects only.%n", projectLimit, projectLimit);
            projectDirs = projectDirs.subList(0, projectLimit);
        }

        ObjectWriter jsonWriter = new ObjectMapper()
                .writer(new DefaultPrettyPrinter().withObjectIndenter(new DefaultIndenter("\t", "\n")));

        int projectsAttempted = 0;
        int projectsWithSummaries = 0;
        int totalClasses = 0;
        int totalSuccesses = 0;
        int totalFailures = 0;
        int totalSkipped = 0;

        try (LlmSummaryWriter csvWriter = new LlmSummaryWriter(LLM_SUMMARY_PATH)) {
            for (File projectDir : projectDirs) {
                projectsAttempted++;
                String relativePath = inputRoot.toPath().relativize(projectDir.toPath()).toString().replace("\\", "/");
                String projectIdentifier = sanitizeProjectIdentifier(relativePath);
                SummaryStats stats = processProject(projectDir, relativePath, projectIdentifier, parseProject, jsonWriter, summaryService, csvWriter);
                totalClasses += stats.getProcessedClasses();
                totalSuccesses += stats.getSuccessfulSummaries();
                totalFailures += stats.getFailedSummaries();
                totalSkipped += stats.getSkippedClasses();
                if (stats.getSuccessfulSummaries() > 0) {
                    projectsWithSummaries++;
                }
            }
        }

        System.out.printf("%nLLM summarisation complete:%n");
        System.out.printf("  Projects attempted: %d%n", projectsAttempted);
        System.out.printf("  Projects with summaries: %d%n", projectsWithSummaries);
        System.out.printf("  Classes processed: %d%n", totalClasses);
        System.out.printf("  Summaries generated: %d%n", totalSuccesses);
        System.out.printf("  Failed summaries: %d%n", totalFailures);
        System.out.printf("  Classes skipped: %d%n", totalSkipped);
        System.out.println("LLM summaries written to " + LLM_SUMMARY_PATH);
    }

    /**
     * Processes a single Java project to generate LLM-based summaries.
     * <p>
     * Parses the project structure, extracts features, generates summaries via LLM,
     * and writes output to both JSON and CSV files.
     * </p>
     * 
     * @param projectDir the project directory to process
     * @param relativePath the relative path from the input root
     * @param projectIdentifier sanitized identifier for file naming
     * @param parseProject the project parser instance
     * @param jsonWriter Jackson writer for JSON output
     * @param summaryService service for generating LLM summaries
     * @param csvWriter CSV writer for summary output
     * @return statistics about the processing results
     */
    private SummaryStats processProject(File projectDir,
                                        String relativePath,
                                        String projectIdentifier,
                                        ParseProject parseProject,
                                        ObjectWriter jsonWriter,
                                        LlmSummaryService summaryService,
                                        LlmSummaryWriter csvWriter) {
        System.out.println();
        System.out.println(relativePath);
        try {
            HashMap<String, Object> parsedProject = parseProject.parseProject(projectDir, relativePath, false);
            if (parsedProject.isEmpty()) {
                System.out.println("  No parseable classes found.");
                return SummaryStats.empty();
            }

            jsonWriter.writeValue(new File("output/json-output/llm/" + projectIdentifier + ".json"), parsedProject);
            SummaryStats stats = summaryService.generateSummaries(parsedProject, projectDir.getName(), projectIdentifier, csvWriter);
            if (stats.hasResults()) {
                System.out.printf("  Project summary: %d classes processed, %d summaries generated, %d failed, %d skipped.%n",
                        stats.getProcessedClasses(),
                        stats.getSuccessfulSummaries(),
                        stats.getFailedSummaries(),
                        stats.getSkippedClasses());
            } else {
                System.out.println("  No eligible classes for LLM summarisation.");
            }
            return stats;
        } catch (IOException e) {
            System.err.println("  IO error while processing " + relativePath + ": " + e.getMessage());
            return SummaryStats.empty();
        } catch (Exception e) {
            System.err.println("  Error while processing " + relativePath + ": " + e.getMessage());
            return SummaryStats.empty();
        }
    }

    /**
     * Creates all required output directories if they don't exist.
     * 
     * @throws IOException if directory creation fails
     */
    private void createDirectories() throws IOException {
        String[] directories = {"output", "output/json-output", "output/json-output/llm", "output/summary-output", "reference"};
        for (String dir : directories) {
            File file = new File(dir);
            if (!file.exists() && !file.mkdirs()) {
                throw new IOException("Unable to create directory: " + dir);
            }
        }
    }

    /**
     * Recursively discovers all directories containing Java files.
     * 
     * @param root the root directory to search
     * @return list of directories containing .java files
     */
    private List<File> findAllProjectDirectories(File root) {
        List<File> projectDirs = new ArrayList<>();
        findProjectDirectoriesRecursive(root, projectDirs);
        return projectDirs;
    }

    /**
     * Recursively searches for directories containing Java files.
     * 
     * @param directory the directory to search
     * @param projectDirs accumulator list for discovered project directories
     */
    private void findProjectDirectoriesRecursive(File directory, List<File> projectDirs) {
        if (!directory.isDirectory()) {
            return;
        }

        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        boolean hasJavaFiles = javaFiles != null && javaFiles.length > 0;

        if (hasJavaFiles) {
            projectDirs.add(directory);
        }

        File[] subdirs = directory.listFiles(File::isDirectory);
        if (subdirs != null) {
            for (File subdir : subdirs) {
                findProjectDirectoriesRecursive(subdir, projectDirs);
            }
        }
    }

    /**
     * Sanitizes a relative path to create a valid filename identifier.
     * 
     * @param relativePath the relative path to sanitize
     * @return sanitized identifier with path separators replaced by underscores
     */
    private String sanitizeProjectIdentifier(String relativePath) {
        return relativePath.replace("/", "_").replace("\\", "_");
    }

    /**
     * Resolves the project limit from configuration.
     * 
     * @param dotEnv configuration map from .env file
     * @return maximum number of projects to process, or 0 for unlimited
     */
    private int resolveProjectLimit(Map<String, String> dotEnv) {
        String raw = resolveValue(dotEnv, "LLM_PROJECT_LIMIT");
        if (raw == null) {
            return 0;
        }
        try {
            int candidate = Integer.parseInt(raw);
            return candidate < 0 ? 0 : candidate;
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid LLM_PROJECT_LIMIT value: " + raw);
            return 0;
        }
    }

    /**
     * Returns the first non-blank value from a trimmed string.
     * 
     * @param value the string to check
     * @return trimmed non-empty string, or null if blank
     */
    private String firstNonBlank(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    /**
     * Resolves a configuration value from .env or environment variables.
     * <p>
     * Checks .env file first, then falls back to OS environment variables.
     * </p>
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @return the resolved value, or null if not found
     */
    private String resolveValue(Map<String, String> dotEnv, String key) {
        String override = dotEnv == null ? null : dotEnv.get(key);
        String value = firstNonBlank(override);
        if (value != null) {
            return value;
        }
        return firstNonBlank(System.getenv(key));
    }

    /**
     * Resolves an integer configuration value with a default fallback.
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @param defaultValue the default value if key not found or invalid
     * @return the resolved integer value
     */
    private int resolveInt(Map<String, String> dotEnv, String key, int defaultValue) {
        String raw = resolveValue(dotEnv, key);
        if (raw == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid " + key + " value: " + raw);
            return defaultValue;
        }
    }

    /**
     * Resolves a double configuration value with a default fallback.
     * 
     * @param dotEnv configuration map from .env file
     * @param key the configuration key to look up
     * @param defaultValue the default value if key not found or invalid
     * @return the resolved double value
     */
    private double resolveDouble(Map<String, String> dotEnv, String key, double defaultValue) {
        String raw = resolveValue(dotEnv, key);
        if (raw == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(raw);
        } catch (NumberFormatException ex) {
            System.out.println("Ignoring invalid " + key + " value: " + raw);
            return defaultValue;
        }
    }
}
