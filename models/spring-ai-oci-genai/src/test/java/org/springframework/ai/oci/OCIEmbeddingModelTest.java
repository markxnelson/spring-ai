/*
 * Copyright 2024 the original author or authors.
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
package org.springframework.ai.oci;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.ai.oci.EmbeddingModelProvider.EMBEDDING_MODEL;

@EnabledIfEnvironmentVariable(named = EmbeddingModelProvider.OCI_COMPARTMENT_ID_KEY, matches = ".+")
public class OCIEmbeddingModelTest {

	private final OCIEmbeddingModel embeddingModel = EmbeddingModelProvider.get();

	@Test
	void embed() {
		List<Double> embedding = embeddingModel.embed(new Document("How many provinces are in Canada?"));
		assertThat(embedding).hasSize(1024);
	}

	@Test
	void call() {
		EmbeddingResponse response = embeddingModel.call(new EmbeddingRequest(
				List.of("How many states are in the USA?", "How many states are in India?"), EmbeddingOptions.EMPTY));
		assertThat(response).isNotNull();
		assertThat(response.getResults()).hasSize(2);
		assertThat(response.getMetadata()).containsEntry("model", EMBEDDING_MODEL);
	}

}
