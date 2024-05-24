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
package org.springframework.ai.autoconfigure.vectorstore.oracle;

import java.util.List;

import javax.sql.DataSource;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mockito;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.context.runner.ContextConsumer;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.PreparedStatementSetter;
import org.springframework.jdbc.core.RowMapper;
import org.testcontainers.junit.jupiter.Testcontainers;

/**
 * @author Fernanda Meheust
 */
@Testcontainers
public class OracleVectorStoreAutoConfigurationIT {

	private static final ContextConsumer<ApplicationContext> testDistanceTypeIsSet = (context -> {
		VectorStore vectorStore = context.getBean(VectorStore.class);
		JdbcTemplate jdbcTemplate = context.getBean(JdbcTemplate.class);
		List<Document> results = vectorStore
			.similaritySearch(SearchRequest.query("What is Great Depression?").withTopK(1));
		String expectedValue = context.getEnvironment().getProperty("expected-value").toString();

		Mockito.verify(jdbcTemplate)
			.query(Mockito.contains(expectedValue), Mockito.any(PreparedStatementSetter.class),
					Mockito.any(RowMapper.class));
	});

	@ParameterizedTest
	@ValueSource(strings = { "COSINE", "DOT", "EUCLIDEAN" })
	public void testAutoConfigDistanceType(String distanceType) {
		ApplicationContextRunner contextRunner = new ApplicationContextRunner()
			.withPropertyValues(String.format("spring.ai.vectorstore.oracle.distanceType=%s", distanceType),
					"spring.ai.vectorstore.oracle.removeExistingVectorStoreTable=false")
			.withUserConfiguration(JdbcTemplateMockApplication.class)
			.withConfiguration(AutoConfigurations.of(OracleVectorStoreAutoConfiguration.class));

		contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);
			JdbcTemplate jdbcTemplate = context.getBean(JdbcTemplate.class);
			List<Document> results = vectorStore
				.similaritySearch(SearchRequest.query("What is Great Depression?").withTopK(1));

			Mockito.verify(jdbcTemplate)
				.query(Mockito.contains(distanceType), Mockito.any(PreparedStatementSetter.class),
						Mockito.any(RowMapper.class));
		});
	}

	@ParameterizedTest
	@CsvSource({ "NONE,NONE", "HNSW,INMEMORY", "IVF,PARTITIONS" })
	public void testAutoConfigIndexType(String indexType, String expectedText) {
		ApplicationContextRunner contextRunner = new ApplicationContextRunner()
			.withPropertyValues(String.format("spring.ai.vectorstore.oracle.indexType=%s", indexType),
					"spring.ai.vectorstore.oracle.distanceType=DOT",
					"spring.ai.vectorstore.oracle.removeExistingVectorStoreTable=false")
			.withUserConfiguration(JdbcTemplateMockApplication.class)
			.withConfiguration(AutoConfigurations.of(OracleVectorStoreAutoConfiguration.class));

		contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);
			JdbcTemplate jdbcTemplate = context.getBean(JdbcTemplate.class);
			List<Document> results = vectorStore
				.similaritySearch(SearchRequest.query("What is Great Depression?").withTopK(1));

			if (expectedText.equals("NONE")) {
				Mockito.verify(jdbcTemplate, Mockito.atMostOnce()).execute(Mockito.anyString());
			}
			else {
				Mockito.verify(jdbcTemplate).execute(Mockito.contains(expectedText));
				Mockito.verify(jdbcTemplate, Mockito.times(2)).execute(Mockito.anyString());
			}
		});
	}

	@Configuration
	public static class JdbcTemplateMockApplication {

		@Bean
		public DataSource myDataSource() {
			return Mockito.mock(DataSource.class);
		}

		@Bean
		public JdbcTemplate myJdbcTemplate(DataSource dataSource) {

			return Mockito.mock(JdbcTemplate.class);
		}

		@Bean
		EmbeddingModel myEmbeddingModel() {
			return Mockito.mock(EmbeddingModel.class);
		}

	}

}
