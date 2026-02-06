<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('analyses', function (Blueprint $table) {
            $table->id();
            $table->string('uuid')->unique();
            $table->string('image_path');
            $table->string('status')->default('pending');
            $table->string('first_en')->nullable();
            $table->integer('first_acc')->nullable();
            $table->string('second_en')->nullable();
            $table->integer('second_acc')->nullable();
            $table->string('third_en')->nullable();
            $table->integer('third_acc')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('analyses');
    }
};
