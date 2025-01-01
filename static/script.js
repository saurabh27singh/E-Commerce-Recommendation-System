const userId = 'user1';
$.ajaxSetup({
    beforeSend: function (xhr, settings) {
        if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type)) {
            xhr.setRequestHeader("X-CSRFToken", document.querySelector('meta[name="csrf-token"]').content);
        }
    }
});

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function viewProduct(productId) {
    const recsContainer = $('#recommendations');
    recsContainer.addClass('loading');

    $.ajax({
        url: '/api/interaction',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            user_id: userId,
            product_id: productId,
            type: 'view'
        }),
        success: function () {
            $.get(`/api/recommendations/hybrid/${userId}/${productId}`)
                .done(function (recommendations) {
                    displayRecommendations(recommendations);
                })
                .fail(function (jqXHR, textStatus, errorThrown) {
                    console.error('Failed to get recommendations:', errorThrown);
                    recsContainer.html('<p>Failed to load recommendations</p>');
                })
                .always(function () {
                    recsContainer.removeClass('loading');
                });
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.error('Failed to record interaction:', errorThrown);
            recsContainer.removeClass('loading');
        }
    });
}

function displayRecommendations(recommendations) {
    const container = $('#recommendations');
    container.empty();

    recommendations.forEach(product => {
        const rating = Number(product.rating);
        const fullStars = Math.floor(rating);
        const hasHalfStar = rating % 1 !== 0;
        const emptyStars = 5 - Math.ceil(rating);

        const stars = '★'.repeat(fullStars) +
            (hasHalfStar ? '½' : '') +
            '☆'.repeat(emptyStars);

        container.append(`
                    <div class="product-card">
                        <h3>${escapeHtml(product.name)}</h3>
                        <p>${escapeHtml(product.description)}</p>
                        <p class="price">$${escapeHtml(String(product.price))}</p>
                        <p class="rating">
                            <span class="rating-stars">${stars}</span>
                            <span>${escapeHtml(String(product.rating))}/5</span>
                        </p>
                        <button onclick="viewProduct(${Number(product.id)})">View Details</button>
                    </div>
                `);
    });
}
