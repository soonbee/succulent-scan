<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Str;

class Analysis extends Model
{
    protected $fillable = [
        'uuid',
        'image_path',
        'status',
        'first_en',
        'first_acc',
        'second_en',
        'second_acc',
        'third_en',
        'third_acc',
    ];

    protected static function boot(): void
    {
        parent::boot();

        static::creating(function (Analysis $analysis): void {
            if (empty($analysis->uuid)) {
                $analysis->uuid = (string) Str::uuid();
            }
        });
    }
}
