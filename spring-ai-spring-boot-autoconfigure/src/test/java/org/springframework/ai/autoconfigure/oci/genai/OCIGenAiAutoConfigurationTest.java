package org.springframework.ai.autoconfigure.oci.genai;

import java.nio.file.Paths;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.oci.OCIEmbeddingModel;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

@EnabledIfEnvironmentVariable(named = OCIGenAiAutoConfigurationTest.COMPARTMENT_ID_KEY, matches = ".+")
public class OCIGenAiAutoConfigurationTest {

	public static final String COMPARTMENT_ID_KEY = "OCI_COMPARTMENT_ID";

	private final String CONFIG_FILE = Paths.get(System.getProperty("user.home"), ".oci", "config").toString();

	private final String COMPARTMENT_ID = System.getenv(COMPARTMENT_ID_KEY);

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner().withPropertyValues(
	// @formatter:off
				"spring.ai.oci.genai.authenticationType=file",
				"spring.ai.oci.genai.file=" + CONFIG_FILE,
				"spring.ai.oci.genai.embedding.compartment=" + COMPARTMENT_ID,
				"spring.ai.oci.genai.embedding.servingMode=on-demand",
				"spring.ai.oci.genai.embedding.model=cohere.embed-english-light-v2.0"
				// @formatter:on
	).withConfiguration(AutoConfigurations.of(OCIGenAiAutoConfiguration.class));

	@Test
	void embeddings() {
		contextRunner.run(context -> {
			OCIEmbeddingModel embeddingModel = context.getBean(OCIEmbeddingModel.class);
			assertThat(embeddingModel).isNotNull();
			EmbeddingResponse response = embeddingModel.call(new EmbeddingRequest(
					List.of("There are 50 states in the USA", "Canada has 10 provinces"), EmbeddingOptions.EMPTY));
			assertThat(response).isNotNull();
			assertThat(response.getResults()).hasSize(2);
		});
	}

}
