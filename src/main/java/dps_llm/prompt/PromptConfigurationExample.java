package dps_llm.prompt;

/**
 * Demonstrates usage of the prompt configuration system.
 * <p>
 * This example shows how to:
 * <ul>
 *   <li>Use default prompts</li>
 *   <li>Switch between different prompt variants</li>
 *   <li>Access prompts directly via PromptManager</li>
 *   <li>List available prompts</li>
 * </ul>
 * </p>
 * 
 * @author Najam
 */
public class PromptConfigurationExample {

    public static void main(String[] args) {
        System.out.println("=== Prompt Configuration System Demo ===\n");
        
        // Example 1: Using default prompt via LlmPromptBuilder
        demonstrateDefaultPrompt();
        
        // Example 2: Using alternative prompts
        demonstrateAlternativePrompts();
        
        // Example 3: Direct access via PromptManager
        demonstrateDirectAccess();
        
        // Example 4: Listing available prompts
        listAvailablePrompts();
    }
    
    private static void demonstrateDefaultPrompt() {
        System.out.println("1. DEFAULT PROMPT (SENIOR_ANALYST_CONCISE)");
        System.out.println("   " + "=".repeat(50));
        
        LlmPromptBuilder builder = new LlmPromptBuilder();
        String systemPrompt = builder.getSystemPrompt();
        
        System.out.println("   " + systemPrompt);
        System.out.println();
    }
    
    private static void demonstrateAlternativePrompts() {
        System.out.println("2. ALTERNATIVE PROMPTS");
        System.out.println("   " + "=".repeat(50));
        
        String[] aliases = {
            "SENIOR_ANALYST_BRIEF",
            "SENIOR_ANALYST_VERBOSE", 
            "SENIOR_ANALYST_DETAILED"
        };
        
        for (String alias : aliases) {
            System.out.println("\n   " + alias + ":");
            LlmPromptBuilder builder = new LlmPromptBuilder(alias);
            String prompt = builder.getSystemPrompt();
            System.out.println("   " + truncate(prompt, 100) + "...");
        }
        System.out.println();
    }
    
    private static void demonstrateDirectAccess() {
        System.out.println("3. DIRECT ACCESS TO OTHER CATEGORIES");
        System.out.println("   " + "=".repeat(50));
        
        PromptManager manager = PromptManager.getInstance();
        
        System.out.println("\n   CODE_REVIEWER:");
        System.out.println("   " + truncate(manager.getPrompt("CODE_REVIEWER"), 100) + "...");
        
        System.out.println("\n   TEST_ENGINEER:");
        System.out.println("   " + truncate(manager.getPrompt("TEST_ENGINEER"), 100) + "...");
        
        System.out.println();
    }
    
    private static void listAvailablePrompts() {
        System.out.println("4. AVAILABLE PROMPT ALIASES");
        System.out.println("   " + "=".repeat(50));
        
        PromptManager manager = PromptManager.getInstance();
        java.util.Set<String> aliases = manager.getAvailableAliases();
        
        System.out.println("   Total prompts loaded: " + aliases.size());
        System.out.println("   Aliases:");
        
        java.util.List<String> sortedAliases = new java.util.ArrayList<>(aliases);
        java.util.Collections.sort(sortedAliases);
        
        for (String alias : sortedAliases) {
            System.out.println("     - " + alias);
        }
        System.out.println();
    }
    
    private static String truncate(String text, int maxLength) {
        if (text.length() <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength);
    }
}
