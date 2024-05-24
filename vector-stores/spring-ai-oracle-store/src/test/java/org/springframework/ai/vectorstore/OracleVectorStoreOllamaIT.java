/*
 * Copyright 2023 - 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.vectorstore;

import static org.junit.Assert.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import javax.sql.DataSource;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.ai.autoconfigure.openai.OpenAiAutoConfiguration;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.ollama.OllamaEmbeddingModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceProperties;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.jdbc.core.JdbcTemplate;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.oracle.OracleContainer;

import com.zaxxer.hikari.HikariDataSource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Corrado De Bari
 * @author Mark Nelson
 */
@Testcontainers
public class OracleVectorStoreOllamaIT {

	private static final int MINILM_EMBEDDING_SIZE = 384;

	private static final String OLLAMA_MODEL = "all-minilm";

	@Container
	static OracleContainer oracleContainer = new OracleContainer("gvenzl/oracle-free:23.4-slim-faststart")
		.withDatabaseName("pdb1")
		.withUsername("testuser")
		.withPassword(("testpwd"));

	@Container
	static OllamaContainer ollama = new OllamaContainer("ollama/ollama:0.1.26").withExposedPorts(11434);

	List<Document> documents = List.of(
			new Document(getText("classpath:/test/data/spring.ai.txt"), Map.of("meta1", "meta1")),
			new Document(getText("classpath:/test/data/time.shelter.txt")),
			new Document(getText("classpath:/test/data/great.depression.txt"), Map.of("meta2", "meta2")));

	public static String getText(String uri) {
		var resource = new DefaultResourceLoader().getResource(uri);
		try {
			return resource.getContentAsString(StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private static void dropTable(ApplicationContext context) {
		JdbcTemplate jdbcTemplate = context.getBean(JdbcTemplate.class);
		jdbcTemplate.execute("drop table vector_store");
	}

	@BeforeAll
	public static void setup() throws Exception {
		// pull a very small model for initial testing
		ollama.execInContainer("ollama", "pull", OLLAMA_MODEL);
	}

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withUserConfiguration(TestApplication.class)
		.withPropertyValues("test.spring.ai.vectorstore.oracle.distanceType=COSINE",

				// JdbcTemplate configuration
				String.format("app.datasource.url=%s", oracleContainer.getJdbcUrl()),
				"app.datasource.username=testuser", "app.datasource.password=testpwd",
				"app.datasource.type=com.zaxxer.hikari.HikariDataSource",
				"spring.ai.ollama.base-url=http://" + ollama.getHost() + ":" + ollama.getMappedPort(11434),
				"spring.ai.ollama.embedding.enabled=true", "spring.ai.ollama.embedding.options.model=" + OLLAMA_MODEL);

	@ParameterizedTest(name = "{0} : {displayName} ")
	@ValueSource(strings = { "COSINE", "DOT" })
	public void simpleEmbeddingTest(String distanceType) throws Exception {
		contextRunner.withPropertyValues("test.spring.ai.vectorstore.oracle.distanceType=" + distanceType)
			.run(context -> {

				VectorStore vectorStore = context.getBean(VectorStore.class);

				vectorStore.add(documents);

				List<Document> results = vectorStore
					.similaritySearch(SearchRequest.query("What is Great Depression").withTopK(1));

				assertThat(results).hasSize(1);
				Document resultDoc = results.get(0);
				assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
				assertThat(resultDoc.getMetadata()).containsKeys("meta2", "distance");

				// Remove all documents from the store
				vectorStore.delete(documents.stream().map(doc -> doc.getId()).toList());

				List<Document> results2 = vectorStore
					.similaritySearch(SearchRequest.query("Great Depression").withTopK(1));
				assertThat(results2).hasSize(0);

				dropTable(context);
			});
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration(exclude = { DataSourceAutoConfiguration.class, OpenAiAutoConfiguration.class })
	public static class TestApplication {

		@Value("${test.spring.ai.vectorstore.oracle.distanceType}")
		OracleVectorStore.OracleDistanceType distanceType;

		@Bean
		public VectorStore vectorStore(JdbcTemplate jdbcTemplate, EmbeddingModel embeddingClient) {
			return new OracleVectorStore(jdbcTemplate, embeddingClient, MINILM_EMBEDDING_SIZE, distanceType, true);
		}

		@Bean
		public JdbcTemplate myJdbcTemplate(DataSource dataSource) {
			return new JdbcTemplate(dataSource);
		}

		@Bean
		@Primary
		@ConfigurationProperties("app.datasource")
		public DataSourceProperties dataSourceProperties() {
			return new DataSourceProperties();
		}

		@Bean
		public HikariDataSource dataSource(DataSourceProperties dataSourceProperties) {
			return dataSourceProperties.initializeDataSourceBuilder().type(HikariDataSource.class).build();
		}

		@Bean
		public EmbeddingModel embeddingClient() {
			var ollamaApi = new OllamaApi("http://" + ollama.getHost() + ":" + ollama.getMappedPort(11434));
			return new OllamaEmbeddingModel(ollamaApi).withModel(OLLAMA_MODEL);
		}

	}

}
